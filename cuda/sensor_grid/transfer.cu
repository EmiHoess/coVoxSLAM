#include "map_native.h"
#include <covoxslam/sensor_grid_overlap.h>
#include <covoxslam/sensor_grid_neighborhood.h>
#include <covoxslam/sensor_grid_fp32.h>
#include "specialized.cuh"
#include "candidates.cuh"
#include "candidate_list.cuh"
#include "../projective/geometry.cuh"

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
template <bool fp32, bool masked = false>
__global__ void transfer_cells(field_view source, interpolation mode, const std::uint32_t* slots,
    const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    unsigned long long* count, float truncation, const std::uint32_t* masks = nullptr,
    unsigned long long* candidate_count = nullptr)
{
    const auto slot = slots[blockIdx.x / 4], local = (blockIdx.x % 4) * 128 + threadIdx.x;
    const auto key = keys[slot];
    const float3 point{(key.x * 8 + int(local % 8) + .5f) * size,
        (key.y * 8 + int(local / 8 % 8) + .5f) * size, (key.z * 8 + int(local / 64) + .5f) * size};
    const auto sensor = detail::projective_reference_point(transform, point);
    bool selected = true;
    if constexpr (masked) selected = (masks[std::size_t(slot) * 16 + local / 32] & (1U << (local % 32))) != 0;
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
        auto value = tiles[slot].voxels[local];
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
         tiles[slot].voxels[local] = value;
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
    std::uint32_t* masks, void* opaque_stream)
{
    if (!count) return 0;

    if (!sampling_detail::valid_view(source) || unsigned(mode) > 2 || !active || !active_count ||
        !slots || !keys || !masks || active_count > 16777216 || count > 1048576)
        return int(cudaErrorInvalidValue);
    const auto bounds = prepare_candidates(source, transform, mode);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    reset_candidates<<<count, 32, 0, stream>>>(slots, masks, !bounds.safe);
    const auto error = cudaGetLastError();

    if (error != cudaSuccess || !bounds.safe) return int(error);
    mark_candidates<<<(active_count + 255) / 256, 256, 0, stream>>>(source, active, active_count,
        bounds, slots, count, keys, masks);
    return int(cudaGetLastError());
}
int cuda_sensor_list_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    candidate_list_view list, void* opaque_stream)
{
    if (!sampling_detail::valid_view(source) || unsigned(mode) > 2 || !list.counters || !list.ids ||
        !list.masks || !list.capacity || list.capacity > 536870912 || !active || !active_count ||
        !slots || !keys || active_count > 16777216 || count > 1048576) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    auto error = cudaMemsetAsync(list.counters, 0, sizeof(candidate_list_counters), stream);

    if (error != cudaSuccess || !count) return int(error);
    const auto bounds = prepare_candidates(source, transform, mode);
    reset_candidates<<<count, 32, 0, stream>>>(slots, list.masks, !bounds.safe);
    error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);

    if (bounds.safe) list_mark<float><<<(active_count + 255) / 256, 256, 0, stream>>>(source, active,
        active_count, bounds, list_boundary_guard(source, bounds), slots, count, keys, list);
    error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);
    compact_candidate_bits<<<count, 32, 0, stream>>>(slots, list);
    return int(cudaGetLastError());
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
