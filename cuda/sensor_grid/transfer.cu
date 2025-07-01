#include "map_native.h"
#include <covoxslam/sensor_grid_overlap.h>
#include <covoxslam/sensor_grid_neighborhood.h>
#include <covoxslam/sensor_grid_fp32.h>
#include "specialized.cuh"
#include "candidates.cuh"
#include "candidate_list.cuh"
#include "coarse_mask.cuh"
#include "resident.h"
#include "tsdf_views.cuh"
#include "../common/geometry.cuh"

namespace covoxslam::sensor_grid
{
namespace
{
__global__ void compact_mask_cells(field_view source, const std::uint32_t* offsets,
    const std::uint32_t* cells, std::uint8_t* flags)
{
    const auto tile = blockIdx.x;
    if (offsets[tile] == offsets[tile + 1])
    {
        if (!threadIdx.x) flags[tile] = 0;
        return;
    }
    bool observed{};

    for (std::size_t i = std::size_t(offsets[tile]) + threadIdx.x;
         i < offsets[tile + 1] && !observed; i += blockDim.x)
    {
        const auto value = source.voxels[cells[i]];
        observed = value.contributions && value.weight > 0 && isfinite(value.weight) && isfinite(value.distance);
    }
    const auto any = __syncthreads_or(observed);

    if (!threadIdx.x) flags[tile] = any ? 1 : 0;
}
__global__ void mask_cells(field_view source, unsigned n, std::uint8_t* flags)
{
    const auto block = blockIdx.x, local = threadIdx.x;
    const auto x = block % n * 8 + local % 8;
    const auto y = block / n % n * 8 + local / 8 % 8;
    const auto z = block / (n * n) * 8 + local / 64;
    bool observed{};

    if (x < source.side && y < source.side && z < source.side) {
        const auto value = source.voxels[x + std::size_t(source.side) * (y + std::size_t(source.side) * z)];
        observed = value.contributions && value.weight > 0 && isfinite(value.weight) && isfinite(value.distance);
    }
    const auto any = __syncthreads_or(observed);
    if (!local) flags[block] = any ? 1 : 0;
}
template <bool fp32, bool masked = false, bool overflow_fallback = false, bool coarse = false,
    bool resident = false, class tile_view = detail::cuda_tsdf_tile*>
__global__ void transfer_cells(field_view source, interpolation mode, const std::uint32_t* slots,
    const detail::cuda_tile_coordinate* keys, tile_view tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    unsigned long long* count, float truncation, const std::uint32_t* masks = nullptr,
    unsigned long long* candidate_count = nullptr, const candidate_list_counters* list_state = nullptr,
    const discovery_state* resident_state = nullptr)
{
    if constexpr (resident)
        if (resident_state->error || blockIdx.x / 4 >= resident_state->count) return;
    if constexpr (overflow_fallback)
        if (!list_state->overflow || list_state->invalid) return;
    const auto slot = slots[blockIdx.x / 4], local = (blockIdx.x % 4) * 128 + threadIdx.x;
    const auto key = keys[slot];
    const float3 point{(key.x * 8 + int(local % 8) + .5f) * size,
        (key.y * 8 + int(local / 8 % 8) + .5f) * size, (key.z * 8 + int(local / 64) + .5f) * size};
    const auto sensor = detail::projective_reference_point(transform, point);
    bool selected = true;
    if constexpr (coarse) selected = (masks[slot] & (1U << (local % 8 / 4 + local / 8 % 8 / 4 * 2 + local / 64 / 4 * 4))) != 0;
    else if constexpr (masked) selected = (masks[std::size_t(slot) * 16 + local / 32] & (1U << (local % 32))) != 0;
    const auto measured_sample = [&]() {
        if (!selected) return sample_value{};
        if constexpr (fp32) return sample_fp32(source, sensor.x, sensor.y, sensor.z, mode);
        return sample(source, sensor.x, sensor.y, sensor.z, mode);
    };
    auto measured = measured_sample();
    if (mode == interpolation::exact_overlap)
    {
        overlap_box box{{sensor.x, sensor.y, sensor.z}, {}, size};
        auto rotation = transform;
        rotation.translation[0] = rotation.translation[1] = rotation.translation[2] = 0;

        for (unsigned axis = 0; axis < 3; ++axis) {
            const auto v = detail::projective_reference_point(rotation, {axis == 0 ? 1.f : 0.f,
                axis == 1 ? 1.f : 0.f, axis == 2 ? 1.f : 0.f});
            box.axes[axis][0] = v.x; box.axes[axis][1] = v.y; box.axes[axis][2] = v.z;
        }
        measured = sample_overlap(source, box);
    }
    else if (mode == interpolation::neighborhood8 || mode == interpolation::neighborhood27)
        measured = sample_neighborhood(source, sensor.x, sensor.y, sensor.z, mode == interpolation::neighborhood8 ? 8 : 27);
    bool updated{};

    if (measured.valid)
    {
        auto value = load_sensor_voxel(tiles, slot, local);
        const float old = value.flags & 1U ? value.weight : 0.f;
        bool use_wide = true;
        if constexpr (fp32) {
            const float total32 = old + measured.weight;
            const float distance32 = (old * (old > 0 ? value.distance : 0.f) +
                measured.weight * measured.distance) / total32;
            if (isfinite(total32) && isfinite(distance32)) {
                value.distance = distance32;
                value.weight = fminf(maximum_weight, total32);
                use_wide = false;
            }
        }
        if (use_wide) {
            const double total = double(old) + measured.weight;
            value.distance = static_cast<float>((double(old) * (old > 0 ? value.distance : 0.f) +
                double(measured.weight) * measured.distance) / total);
            value.weight = static_cast<float>(fmin(double(maximum_weight), total));
        }
        if constexpr (fp32)
            value.distance = fmaxf(-truncation, fminf(truncation, value.distance));

        value.flags = 1U | (fabsf(value.distance) <= band ? 2U : 0U);
        if (value.sample_count != UINT32_MAX) ++value.sample_count;
         store_sensor_voxel(tiles, slot, local, value);
        updated = true;
    }
    const unsigned updates = __syncthreads_count(updated);
    if (!threadIdx.x && updates) atomicAdd(count, static_cast<unsigned long long>(updates));
    if constexpr (masked) {
        const unsigned candidates = __syncthreads_count(selected);

        if (!threadIdx.x) atomicAdd(candidate_count, static_cast<unsigned long long>(candidates));
    }
}
}
int cuda_sensor_resident_soa(field_view source, resident_discovery_view view,
    storage::device_descriptor tiles, detail::cuda_projective_pose transform, float size,
    float maximum_weight, float band, unsigned long long* updates, void* stream)
{
    if (!valid_resident_discovery(view) || !sampling_detail::valid_view(source) ||
        tiles.record_schema != storage::schema::tsdf || tiles.storage_layout != storage::layout::pool_soa ||
        tiles.size_tiles < view.scan.capacity || tiles.size_tiles > tiles.capacity_tiles || !tiles.generation ||
        tiles.tile_stride != 2048 || tiles.voxel_stride != 4 || !updates ||
        !(size > 0) || !isfinite(size) || !(maximum_weight > 0) || !isfinite(maximum_weight) ||
        !(band >= 0) || !isfinite(band)) return int(cudaErrorInvalidValue);

    for (unsigned i = 0; i < 4; ++i)
        if (!tiles.fields[i]) return int(cudaErrorInvalidValue);
    sensor_tsdf_view target{storage::device_accessor<storage::schema::tsdf>{tiles}};
    transfer_cells<false, false, false, false, true><<<view.scan.capacity * 4, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        source, interpolation::nearest, view.slots, view.keys, target, transform, size, maximum_weight,
        band, updates, FLT_MAX, nullptr, nullptr, nullptr, view.scan.state);
    return int(cudaGetLastError());
}
int cuda_sensor_resident_transfer(field_view source, interpolation mode, resident_discovery_view view,
    detail::cuda_tsdf_tile* tiles, detail::cuda_projective_pose transform, float size,
    float maximum_weight, float band, unsigned long long* updates, void* stream)
{
    if (!valid_resident_discovery(view) || !sampling_detail::valid_view(source) || unsigned(mode) > 5 ||
        !tiles || !updates || !(size > 0) || !isfinite(size) || !(maximum_weight > 0) ||
        !isfinite(maximum_weight) || !(band >= 0) || !isfinite(band)) return int(cudaErrorInvalidValue);
    transfer_cells<false, false, false, false, true><<<view.scan.capacity * 4, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        source, mode, view.slots, view.keys, tiles, transform, size, maximum_weight, band,
        updates, FLT_MAX, nullptr, nullptr, nullptr, view.scan.state);
    return int(cudaGetLastError());
}
int cuda_sensor_coarse_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    std::uint32_t* masks, void* opaque_stream, unsigned* launches)
{
    if (launches) *launches = 0;
    if (!count) return 0;

    if (!sampling_detail::valid_view(source) || unsigned(mode) > 2 || !active || !active_count ||
        !slots || !keys || !masks || active_count > 16777216 || count > 1048576)
        return int(cudaErrorInvalidValue);
    const auto bounds = prepare_candidates(source, transform, mode);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    if (launches) *launches = bounds.safe ? 2 : 1;
    reset_coarse_candidates<<<count, 32, 0, stream>>>(slots, masks, !bounds.safe);
    const auto error = cudaGetLastError();

    if (error != cudaSuccess || !bounds.safe) return int(error);
    mark_coarse_candidates<<<(active_count + 255) / 256, 256, 0, stream>>>(source, active, active_count,
        bounds, slots, count, keys, masks);
    return int(cudaGetLastError());
}
int cuda_sensor_coarse_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    const std::uint32_t* masks, unsigned long long* updates, unsigned long long* candidates, void* opaque_stream)
{
    if (!count) return 0;

    if (!sampling_detail::valid_view(source) || unsigned(mode) > 2 || !slots || !keys || !tiles ||
        !masks || !updates || !candidates || count > 1048576 || !(size > 0) || !isfinite(size) ||
        !(maximum_weight > 0) || !isfinite(maximum_weight) || !(band >= 0) || !isfinite(band))
        return int(cudaErrorInvalidValue);
    transfer_cells<false, true, false, true><<<count * 4, 128, 0, static_cast<cudaStream_t>(opaque_stream)>>>(
        source, mode, slots, keys, tiles, transform, size, maximum_weight, band, updates, FLT_MAX, masks, candidates);
    return int(cudaGetLastError());
}
int cuda_sensor_masked_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    const std::uint32_t* masks, unsigned long long* updates, unsigned long long* candidates,
    void* opaque_stream, bool compacted_control)
{
    if (!count) return 0;

    if (!sampling_detail::valid_view(source) || unsigned(mode) > 2 || !slots || !keys || !tiles ||
        !masks || !updates || !candidates || !(size > 0) || !isfinite(size) ||
        !(maximum_weight > 0) || !isfinite(maximum_weight) || !(band >= 0) || !isfinite(band))
        return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);

    if (compacted_control)
        compacted_candidate_transfer<<<count, 32, 0, stream>>>(source, mode, slots, keys, masks, tiles,
            transform, size, maximum_weight, band, updates, candidates);
    else
        transfer_cells<false, true><<<count * 4, 128, 0, stream>>>(source, mode, slots, keys, tiles,
            transform, size, maximum_weight, band, updates, FLT_MAX, masks, candidates);
    return int(cudaGetLastError());
}
int cuda_sensor_list_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    candidate_list_view list, unsigned long long* updates, unsigned long long* candidates, void* opaque_stream)
{
    if (!count) return 0;

    if (!sampling_detail::valid_view(source) || unsigned(mode) > 2 || !slots || !keys || !tiles || count > 1048576 ||
        !list.masks || !list.ids || !list.counters || !list.capacity || list.capacity > 536870912 ||
        !updates || !candidates || !(size > 0) || !isfinite(size) || !(maximum_weight > 0) ||
        !isfinite(maximum_weight) || !(band >= 0) || !isfinite(band)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    const auto extent = list.capacity < count * 512 ? list.capacity : count * 512;
    transfer_candidate_list<<<(extent + 127) / 128, 128, 0, stream>>>(source, mode, list, keys, tiles,
        transform, size, maximum_weight, band, updates, candidates);
    const auto error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);
    transfer_cells<false, true, true><<<count * 4, 128, 0, stream>>>(source, mode, slots, keys, tiles,
        transform, size, maximum_weight, band, updates, FLT_MAX, list.masks, candidates, list.counters);
    return int(cudaGetLastError());
}
int cuda_sensor_mask(field_view source, std::uint8_t* flags, void* stream)
{
    if (!source.voxels || !source.side || source.side > 1624 || !flags) return int(cudaErrorInvalidValue);
    const auto n = (source.side + 7) / 8;
    mask_cells<<<n * n * n, 512, 0, static_cast<cudaStream_t>(stream)>>>(source, n, flags);
    return int(cudaGetLastError());
}
int cuda_sensor_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    std::uint32_t* masks, void* opaque_stream, unsigned* launches)
{
    if (launches) *launches = 0;
    if (!count) return 0;

    if (!sampling_detail::valid_view(source) || unsigned(mode) > 2 || !active || !active_count ||
        !slots || !keys || !masks || active_count > 16777216 || count > 1048576)
        return int(cudaErrorInvalidValue);
    const auto bounds = prepare_candidates(source, transform, mode);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    if (launches) *launches = bounds.safe ? 2 : 1;
    reset_candidates<<<count, 32, 0, stream>>>(slots, masks, !bounds.safe);
    const auto error = cudaGetLastError();

    if (error != cudaSuccess || !bounds.safe) return int(error);
    mark_candidates<<<(active_count + 255) / 256, 256, 0, stream>>>(source, active, active_count,
        bounds, slots, count, keys, masks);
    return int(cudaGetLastError());
}
template<bool append, class real = float>
int launch_list_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    candidate_list_view list, void* opaque_stream, unsigned* launches)
{
    if (launches) *launches = 0;
    if (!sampling_detail::valid_view(source) || unsigned(mode) > 2 || !list.counters || !list.ids ||
        !list.masks || !list.capacity || list.capacity > 536870912 || !active || !active_count ||
        !slots || !keys || active_count > 16777216 || count > 1048576) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    auto error = cudaMemsetAsync(list.counters, 0, sizeof(candidate_list_counters), stream);

    if (error != cudaSuccess || !count) return int(error);
    const auto bounds = prepare_candidates(source, transform, mode);
    if (launches) *launches = 2 + bounds.safe + (!append || !bounds.safe);
    reset_candidates<<<count, 32, 0, stream>>>(slots, list.masks, !bounds.safe);
    error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);

    if (bounds.safe) list_mark<real, append><<<(active_count + 255) / 256, 256, 0, stream>>>(source, active,
        active_count, bounds, list_boundary_guard(source, bounds), slots, count, keys, list);
    error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);
    if (!append || !bounds.safe) compact_candidate_bits<<<count, 32, 0, stream>>>(slots, list);
    error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);
    validate_candidate_list<<<(count + 127) / 128, 128, 0, stream>>>(slots, count, keys, list.counters);
    return int(cudaGetLastError());
}
int cuda_sensor_list_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    candidate_list_view list, void* stream, unsigned* launches)
{
    return launch_list_candidates<false>(source, mode, active, active_count, slots, count, keys, transform, list, stream, launches);
}
int cuda_sensor_append_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    candidate_list_view list, void* stream, unsigned* launches)
{
    return launch_list_candidates<true>(source, mode, active, active_count, slots, count, keys, transform, list, stream, launches);
}
int cuda_sensor_list_candidates_fp64(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    candidate_list_view list, void* stream, unsigned* launches)
{
    return launch_list_candidates<false, double>(source, mode, active, active_count, slots, count, keys, transform, list, stream, launches);
}
int cuda_sensor_append_candidates_fp64(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    candidate_list_view list, void* stream, unsigned* launches)
{
    return launch_list_candidates<true, double>(source, mode, active, active_count, slots, count, keys,
        transform, list, stream, launches);
}
int cuda_sensor_compact_mask(field_view source, const std::uint32_t* offsets,
    const std::uint32_t* cells, std::uint8_t* flags, void* stream)
{
    if (!source.voxels || !source.side || source.side > 1624 || !offsets || !cells || !flags)
        return int(cudaErrorInvalidValue);
    const auto n = (source.side + 7) / 8;
    compact_mask_cells<<<n * n * n, 128, 0, static_cast<cudaStream_t>(stream)>>>(source, offsets, cells, flags);
    return int(cudaGetLastError());
}
int cuda_sensor_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    unsigned long long* updates, void* stream, bool fp32, bool specialized, float truncation)
{
    if (!count) return int(cudaSuccess);
    if (fp32 && (!(truncation > 0) || !isfinite(truncation))) return int(cudaErrorInvalidValue);
    if (specialized && (fp32 || mode != interpolation::trilinear)) return int(cudaErrorInvalidValue);
    if (!sampling_detail::valid_view(source) || !slots || !keys || !tiles || !updates || unsigned(mode) > 5 || (fp32 && unsigned(mode) > 2) ||
        !(size > 0) || !isfinite(size) || !(maximum_weight > 0) || !isfinite(maximum_weight) ||
        !(band >= 0) || !isfinite(band)) return int(cudaErrorInvalidValue);
    if (specialized) {
        double norm{};

        for (float value : transform.rotation_xyzw) {
            if (!std::isfinite(value)) return int(cudaErrorInvalidValue);
            norm += double(value) * value;
        }

        for (float value : transform.translation)
            if (!std::isfinite(value)) return int(cudaErrorInvalidValue);

        if (std::abs(norm - 1) > .001) return int(cudaErrorInvalidValue);
    }
    if (specialized) specialized_transfer<<<count, 128, 0, static_cast<cudaStream_t>(stream)>>>(source, slots, keys, tiles,
        transform, prepare_affine(transform, size), size, maximum_weight, band, updates);
    else if (fp32) transfer_cells<true><<<count * 4, 128, 0, static_cast<cudaStream_t>(stream)>>>(source, mode, slots, keys, tiles,
        transform, size, maximum_weight, band, updates, truncation);
    else transfer_cells<false><<<count * 4, 128, 0, static_cast<cudaStream_t>(stream)>>>(source, mode, slots, keys, tiles,
        transform, size, maximum_weight, band, updates, truncation);
    return int(cudaGetLastError());
}
}
