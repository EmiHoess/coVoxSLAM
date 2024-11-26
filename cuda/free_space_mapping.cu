#include "ray_resident_free_space_skip/skip.cuh"
#include "ray_resident_warp/dda.cuh"
#include <cmath>

namespace covoxslam::detail
{
namespace
{
__global__ void begin_resident_skip(cuda_resident_view view)
{
    if (view.state->error) return;

    if (!view.state->batch_open || view.state->batch_task_phase || view.state->batch_contributions ||
        view.state->batch_reduction != static_cast<unsigned>(cuda_resident_reduction::direct_atomic)) {
        view.state->error = resident_scatter_error;
        return;
    }
    view.state->batch_task_phase = 7;
}

__global__ void walk_resident_free_space(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_free_space_view skip)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= count || atomicAdd(&view.state->error, 0U)) return;
    auto walk = resident_warp_begin(view, rays[index], config);
    unsigned skipped{};
    resident_warp_counters counters{};

    while (walk.active)
    {
        unsigned voxel{};
        double distance{}, weight{};
         float raw_distance{};

        if (!resident_warp_sample(view, walk, config, voxel, distance, weight, &raw_distance)) continue;
        if (resident_skip_free_space(skip, voxel, raw_distance, config.truncation_distance)) ++skipped;
        else resident_warp_flush(view, reduction, voxel, distance, weight, 1, counters);
    }
    atomicAdd(walk.contributions ? &view.state->batch_integrated : &view.state->batch_rejected, 1U);
    atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_contributions),
        static_cast<unsigned long long>(walk.contributions - skipped));
    if (skipped) atomicAdd(&view.state->batch_skipped, static_cast<unsigned long long>(skipped));
}
}

int cuda_walk_resident_skip(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_free_space_view skip, void* stream) noexcept
{
    if (!skip.flags || !std::isfinite(skip.minimum_weight) || skip.minimum_weight <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto discovered = cuda_walk_resident_reduction(view, reduction, rays, count, config,
        cuda_resident_reduction::direct_atomic, 0, stream);

    if (discovered) return discovered;
    const auto queue = static_cast<cudaStream_t>(stream);
    begin_resident_skip<<<1, 1, 0, queue>>>(view);
    const auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    if (count) walk_resident_free_space<<<static_cast<unsigned>((count + 255) / 256), 256, 0, queue>>>(view, reduction, rays, count, config, skip);
    return static_cast<int>(cudaGetLastError());
}
}
