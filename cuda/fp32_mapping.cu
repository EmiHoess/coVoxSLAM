#include "ray_resident_free_space_skip_fp32/fusion.cuh"
#include "ray_resident_free_space_skip/skip.cuh"
#include "ray_resident_warp/dda.cuh"
#include <cmath>

namespace covoxslam::detail
{
namespace
{
__global__ void fp32_phase(cuda_resident_view view, unsigned stage, unsigned rays)
{
    auto& state = *view.state;
    if (state.error) return;

    if (stage ? !state.batch_open || state.batch_task_phase != stage + 12 || state.batch_compact_rays != rays : state.batch_open) {
        state.error = resident_scatter_error;
        return;
    }
    if (!stage) {
        state.batch_open = 1;
        state.batch_compact_rays = rays;
        state.batch_contributions = state.batch_touched = state.batch_validated = state.batch_skipped = 0;
        state.batch_integrated = state.batch_rejected = 0;
         state.batch_reduction = static_cast<unsigned>(cuda_resident_reduction::direct_atomic);
    }
    state.batch_task_phase = stage + 13;
}

__global__ void clear_fp32(cuda_resident_view view, cuda_fp32_view sums)
{
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= std::size_t(view.tile_capacity) * 512 || view.state->error) return;
    view.counts[index] = view.cursors[index] = 0;
     sums.weights[index] = sums.weighted_distances[index] = 0;
}

__global__ void walk_fp32(cuda_resident_view view, cuda_fp32_view sums, const cuda_device_ray* rays,
    unsigned count, cuda_ray_parameters config, cuda_free_space_view skip, bool discovery)
{
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count || atomicAdd(&view.state->error, 0U)) return;
    auto walk = resident_warp_begin(view, rays[index], config);
    unsigned skipped{};

    while (walk.active) {
        unsigned voxel{};
        double weighted{}, weight{};
        float distance{};

        if (discovery) { resident_warp_sample<true>(view, walk, config, voxel, weighted, weight); continue; }
        if (!resident_warp_sample(view, walk, config, voxel, weighted, weight, &distance)) continue;
        if (resident_skip_free_space(skip, voxel, distance, config.truncation_distance)) ++skipped;
        else resident_fp32_add(view, sums, voxel, fminf(config.truncation_distance,
            fmaxf(-config.truncation_distance, distance)), static_cast<float>(weight));
    }
    if (discovery) return;
    atomicAdd(walk.contributions ? &view.state->batch_integrated : &view.state->batch_rejected, 1U);
    atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_contributions),
        static_cast<unsigned long long>(walk.contributions - skipped));
    atomicAdd(&view.state->batch_skipped, static_cast<unsigned long long>(skipped));
}

__global__ void finish_fp32(cuda_resident_view view, bool validate)
{
    auto& state = *view.state;

    if (state.error) return;
    if (state.batch_validated != state.batch_contributions ||
        std::uint64_t(state.batch_integrated) + state.batch_rejected != state.batch_compact_rays) {
        state.error = resident_scatter_error;
        return;
    }
    if (validate) return;
    state.contributions += state.batch_contributions;
    state.skipped += state.batch_skipped;
    state.touched += state.batch_touched;
    state.integrated += state.batch_integrated;
     state.rejected += state.batch_rejected;
    ++state.batches;
    state.batch_open = 0;
}
}

int cuda_resident_fp32_stage(cuda_resident_view view, cuda_fp32_view sums,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_fusion_parameters fusion,
    cuda_free_space_view skip, unsigned stage, void* stream) noexcept
{
    if (!valid_resident_map(view) || !sums.weights || !sums.weighted_distances || !skip.flags || stage > 3 ||
        !std::isfinite(skip.minimum_weight) || skip.minimum_weight <= 0 || count > UINT32_MAX || (count && !rays) ||
        !std::isfinite(config.voxel_size) || config.voxel_size <= 0 || !std::isfinite(config.truncation_distance) || config.truncation_distance <= 0 ||
        !std::isfinite(config.minimum_ray_length_m) || config.minimum_ray_length_m <= 0 ||
        !std::isfinite(config.maximum_ray_length_m) || config.maximum_ray_length_m <= config.minimum_ray_length_m ||
        (double(config.maximum_ray_length_m) + config.truncation_distance) / config.voxel_size > (UINT32_MAX - 16.0) / 3 ||
        fusion.truncation_distance != config.truncation_distance || !std::isfinite(fusion.maximum_weight) || fusion.maximum_weight <= 0 ||
        !std::isfinite(fusion.extraction_distance) || fusion.extraction_distance < 0 || fusion.extraction_distance > fusion.truncation_distance)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto queue = static_cast<cudaStream_t>(stream);
    const auto blocks = view.tile_capacity * 2;
    fp32_phase<<<1, 1, 0, queue>>>(view, stage, static_cast<unsigned>(count));
    auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    if (!stage) {
        clear_fp32<<<blocks, 256, 0, queue>>>(view, sums);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    if (stage < 2) {
        if (count) walk_fp32<<<static_cast<unsigned>((count + 255) / 256), 256, 0, queue>>>(view, sums, rays, static_cast<unsigned>(count), config, skip, !stage);
    }
    else if (stage == 2)
    {
        prepare_resident_fp32<<<blocks, 256, 0, queue>>>(view, sums, fusion);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        finish_fp32<<<1, 1, 0, queue>>>(view, true);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        publish_resident_free_space<<<blocks, 256, 0, queue>>>(view, skip, config.truncation_distance);
    }
    else finish_fp32<<<1, 1, 0, queue>>>(view, false);
    return static_cast<int>(cudaGetLastError());
}
}
