#include "ray_resident_compact/discovery.cuh"
#include "ray_resident_free_space_skip/skip.cuh"
#include "ray_resident_reduce/common.cuh"
#include "ray_resident_compact_batched/bookkeeping.cuh"
#include <cmath>

namespace covoxslam::detail
{
namespace
{
__global__ void compact_phase(cuda_resident_view view, unsigned stage, unsigned rays)
{
    auto& state = *view.state;
    if (state.error) return;

    if (stage ? !state.batch_open || state.batch_task_phase != stage + 8 || state.batch_compact_rays != rays : state.batch_open) {
        state.error = resident_scatter_error;
        return;
    }
    if (!stage)
    {
        state.batch_open = 1;
        state.batch_compact_rays = rays;
        state.batch_contributions = state.batch_touched = state.batch_validated = state.batch_skipped = 0;
         state.batch_integrated = state.batch_rejected = 0;
        state.batch_reduction = static_cast<unsigned>(cuda_resident_reduction::direct_atomic);

        for (auto& count : state.batch_warp_counts) count = 0;
    }
    state.batch_task_phase = stage + 9;
}

template<bool batched>
__global__ void compact_rays(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, unsigned count, cuda_ray_parameters config,
    cuda_free_space_view skip, cuda_compact_view compact, unsigned stage)
{
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= count || atomicAdd(&view.state->error, 0U)) return;
    if (!stage) { resident_compact_discover(view, rays[index], config); return; }
    auto walk = resident_warp_begin(view, rays[index], config);
    unsigned skipped{};

    while (walk.active)
    {
        unsigned voxel{};
        double weighted{}, weight{};
        float distance{};

        if (!resident_warp_sample(view, walk, config, voxel, weighted, weight, &distance)) continue;
        if (resident_skip_free_space(skip, voxel, distance, config.truncation_distance)) ++skipped;
        else if constexpr (batched) compact_batched_add(view, reduction, compact, voxel, weighted, weight);
        else resident_compact_add(view, reduction, compact, voxel, weighted, weight);
    }
    atomicAdd(walk.contributions ? &view.state->batch_integrated : &view.state->batch_rejected, 1U);
    atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_contributions),
        static_cast<unsigned long long>(walk.contributions - skipped));
     atomicAdd(&view.state->batch_skipped, static_cast<unsigned long long>(skipped));
}

template<unsigned stage>
__global__ void compact_voxels(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_compact_view compact, cuda_fusion_parameters config, cuda_free_space_view skip)
{
    const auto capacity = std::uint64_t(view.tile_capacity) * 512;
    const auto count = view.state->batch_touched < capacity ? view.state->batch_touched : capacity;
    unsigned maximum{};

    for (std::uint64_t index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
        index < count; index += std::uint64_t(gridDim.x) * blockDim.x)
    {
        if constexpr (stage < 2) { if (atomicAdd(&view.state->error, 0U)) return; }
        const auto voxel = compact.touched[index];

        if (voxel >= capacity) { atomicOr(&view.state->error, resident_scatter_error); continue; }
        const auto observations = view.counts[voxel];
        if constexpr (stage == 0)
        {
            if (!observations || view.cursors[voxel] != observations || voxel / 512 >= view.state->tile_count ||
                view.state->tile_count > view.tile_capacity) {
                atomicOr(&view.state->error, resident_scatter_error);
                continue;
            }
            auto value = view.tiles[voxel / 512].voxels[voxel % 512];

            if (!finish_resident_reduction(value, reduction.weighted_distances[voxel], reduction.weights[voxel], observations, config))
                atomicOr(&view.state->error, resident_numeric_error);
            else view.pending[voxel] = value;
            atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_validated), static_cast<unsigned long long>(observations));
        }
        else if constexpr (stage == 1 || stage == 3) {
            if (!view.state->error) {
            const auto value = view.pending[voxel];
            view.tiles[voxel / 512].voxels[voxel % 512] = value;
             skip.flags[voxel] = resident_free_space_eligible(value, config.truncation_distance, skip.minimum_weight);
            if constexpr (stage == 1) atomicMax(&view.state->maximum_segment_length, observations);
            else maximum = max(maximum, observations);
            }
        }
        if constexpr (stage >= 2)
        {
            view.counts[voxel] = view.cursors[voxel] = 0;
            reduction.weights[voxel] = reduction.weighted_distances[voxel] = 0;
        }
    }
    if constexpr (stage == 3) {

        for (unsigned offset = 16; offset; offset /= 2)
            maximum = max(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
        if (!(threadIdx.x & 31) && maximum) atomicMax(&view.state->maximum_segment_length, maximum);
    }
}

__global__ void compact_finish(cuda_resident_view view, bool publish)
{
    auto& state = *view.state;

    if (state.error) return;
    if (state.batch_validated != state.batch_contributions || state.batch_touched > std::uint64_t(view.tile_capacity) * 512 ||
        std::uint64_t(state.batch_integrated) + state.batch_rejected != state.batch_compact_rays) {
        state.error = resident_scatter_error;
        return;
    }
    if (publish) return;
    state.contributions += state.batch_contributions;
    state.skipped += state.batch_skipped;
    state.touched += state.batch_touched;
     state.integrated += state.batch_integrated;
    state.rejected += state.batch_rejected;
    ++state.batches;
    state.batch_open = 0;
}
}

template<bool batched>
int compact_stage(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_fusion_parameters fusion,
    cuda_free_space_view skip, cuda_compact_view compact, unsigned stage, void* stream) noexcept
{
    if (!valid_resident_reduction_view(view, reduction, cuda_resident_reduction::direct_atomic) || !compact.touched || !skip.flags ||
        !std::isfinite(skip.minimum_weight) || skip.minimum_weight <= 0 || stage > 3 || count > UINT32_MAX || (count && !rays) ||
        !std::isfinite(config.voxel_size) || config.voxel_size <= 0 ||
        !std::isfinite(config.truncation_distance) || config.truncation_distance <= 0 ||
        !std::isfinite(config.minimum_ray_length_m) || config.minimum_ray_length_m <= 0 ||
        !std::isfinite(config.maximum_ray_length_m) || config.maximum_ray_length_m <= config.minimum_ray_length_m ||
        (double(config.maximum_ray_length_m) + config.truncation_distance) / config.voxel_size > (UINT32_MAX - 16.0) / 3 ||
        fusion.truncation_distance != config.truncation_distance || !std::isfinite(fusion.maximum_weight) || fusion.maximum_weight <= 0 ||
        !std::isfinite(fusion.extraction_distance) || fusion.extraction_distance < 0 || fusion.extraction_distance > fusion.truncation_distance)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto queue = static_cast<cudaStream_t>(stream);
    const auto blocks = static_cast<unsigned>(std::fmin(65535.0, double(view.tile_capacity) * 2));
    compact_phase<<<1, 1, 0, queue>>>(view, stage, static_cast<unsigned>(count));
    auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    if (stage < 2)
    {
        if (count) compact_rays<batched><<<static_cast<unsigned>((count + 255) / 256), 256, 0, queue>>>(
            view, reduction, rays, static_cast<unsigned>(count), config, skip, compact, stage);
    }
    else if (stage == 2) {
        compact_voxels<0><<<blocks, 256, 0, queue>>>(view, reduction, compact, fusion, skip);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        compact_finish<<<1, 1, 0, queue>>>(view, true);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        if constexpr (!batched) compact_voxels<1><<<blocks, 256, 0, queue>>>(view, reduction, compact, fusion, skip);
    }
    else
    {
        compact_voxels<batched ? 3 : 2><<<blocks, 256, 0, queue>>>(view, reduction, compact, fusion, skip);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        compact_finish<<<1, 1, 0, queue>>>(view, false);
    }
    return static_cast<int>(cudaGetLastError());
}

int cuda_resident_compact_stage(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_fusion_parameters fusion,
    cuda_free_space_view skip, cuda_compact_view compact, unsigned stage, void* stream, bool batched) noexcept
{
    return batched ? compact_stage<true>(view, reduction, rays, count, config, fusion, skip, compact, stage, stream) :
         compact_stage<false>(view, reduction, rays, count, config, fusion, skip, compact, stage, stream);
}
}
