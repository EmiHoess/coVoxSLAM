#include "ray_resident_segmented/types.h"
#include "ray_resident_warp/dda.cuh"
#include "ray_resident_free_space_skip/skip.cuh"

namespace covoxslam::detail
{
namespace
{
__global__ void segment_phase(cuda_resident_view view, unsigned previous, unsigned next)
{
    if (view.state->error) return;

    if (!view.state->batch_open || view.state->batch_task_phase != previous ||
        view.state->batch_reduction != static_cast<unsigned>(cuda_resident_reduction::direct_atomic))
        view.state->error = resident_scatter_error;
    else view.state->batch_task_phase = next;
}

__global__ void discover_segments(cuda_resident_view view, const cuda_device_ray* rays,
    unsigned count, cuda_ray_parameters config, cuda_segment_view segments, float length)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= count || atomicAdd(&view.state->error, 0U)) return;
    auto* checkpoints = segments.checkpoints + std::size_t(index) * segments.slots;

    for (unsigned slot = 0; slot < segments.slots; ++slot) checkpoints[slot].visits = 0;
    segments.accepted[index] = 0;
    auto walk = resident_warp_begin(view, rays[index], config);
    segments.prepared[index] = walk;

    while (walk.active)
    {
        const double relative = fmax(0.0, walk.entered - walk.start);
        const double owner = floor(relative / length);

        if (!(owner < segments.slots)) { atomicOr(&view.state->error, resident_scatter_error); return; }
        auto& checkpoint = checkpoints[static_cast<unsigned>(owner)];

        if (!checkpoint.visits) {
            for (int axis = 0; axis < 3; ++axis) {
                checkpoint.voxel[axis] = walk.voxel[axis];
                 checkpoint.next[axis] = walk.next[axis];
            }
            checkpoint.entered = walk.entered;
            checkpoint.iteration = walk.iteration;
        }
        ++checkpoint.visits;
        unsigned voxel{};
        double distance{}, weight{};
        resident_warp_sample<true>(view, walk, config, voxel, distance, weight);
    }
}

__global__ void accumulate_segments(cuda_resident_view view, cuda_resident_reduction_view reduction,
    unsigned count, cuda_ray_parameters config, cuda_free_space_view skip, cuda_segment_view segments, unsigned stride)
{
    const auto size = std::uint64_t(count) * segments.slots;

    for (auto task = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; task < size; task += std::uint64_t(gridDim.x) * blockDim.x)
    {
        if (atomicAdd(&view.state->error, 0U)) return;
        const auto scheduled = resident_segment_at(task, count, segments.slots, stride);
        const auto& checkpoint = segments.checkpoints[std::size_t(scheduled.ray) * segments.slots + scheduled.segment];
        if (!checkpoint.visits) continue;
        auto walk = segments.prepared[scheduled.ray];

        for (int axis = 0; axis < 3; ++axis) {
            walk.voxel[axis] = checkpoint.voxel[axis];
             walk.next[axis] = checkpoint.next[axis];
        }
        walk.entered = checkpoint.entered;
        walk.iteration = checkpoint.iteration;
        unsigned skipped{};
        resident_warp_counters counters{};

        for (unsigned visit = 0; visit < checkpoint.visits; ++visit)
        {
            unsigned voxel{};
            double distance{}, weight{};
            float raw{};

            if (!walk.active) { atomicOr(&view.state->error, resident_scatter_error); return; }
            if (!resident_warp_sample(view, walk, config, voxel, distance, weight, &raw)) continue;
            if (resident_skip_free_space(skip, voxel, raw, config.truncation_distance)) ++skipped;
            else resident_warp_flush(view, reduction, voxel, distance, weight, 1, counters);
        }
        atomicAdd(segments.accepted + scheduled.ray, walk.contributions);
        atomicAdd(reinterpret_cast<unsigned long long*>(&view.state->batch_contributions),
            static_cast<unsigned long long>(walk.contributions - skipped));
        if (skipped) atomicAdd(&view.state->batch_skipped, static_cast<unsigned long long>(skipped));
    }
}

__global__ void count_segment_rays(cuda_resident_view view, cuda_segment_view segments, unsigned count)
{
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count && !view.state->error)
        atomicAdd(segments.accepted[index] ? &view.state->batch_integrated : &view.state->batch_rejected, 1U);
}
}

int cuda_walk_resident_segments(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_free_space_view skip,
    cuda_segment_view segments, float length, void* stream) noexcept
{
    const auto slots = resident_segment_slots(config.maximum_ray_length_m, config.truncation_distance, length);

    if (!valid_resident_reduction_view(view, reduction, cuda_resident_reduction::direct_atomic) ||
        !skip.flags || !std::isfinite(skip.minimum_weight) || skip.minimum_weight <= 0 || !slots || slots != segments.slots ||
        !segments.prepared || !segments.checkpoints || !segments.accepted || !segments.ray_capacity ||
        count > segments.ray_capacity || (count && !rays) || !std::isfinite(config.voxel_size) || config.voxel_size <= 0 ||
        !std::isfinite(config.minimum_ray_length_m) || config.minimum_ray_length_m <= 0 ||
        config.minimum_ray_length_m > config.maximum_ray_length_m ||
        (double(config.maximum_ray_length_m) + config.truncation_distance) / config.voxel_size >= 10000000)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto queue = static_cast<cudaStream_t>(stream);
    segment_phase<<<1, 1, 0, queue>>>(view, 0, 8);
    auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    if (count) {
        const auto blocks = static_cast<unsigned>((count + 255) / 256);
        discover_segments<<<blocks, 256, 0, queue>>>(view, rays, static_cast<unsigned>(count), config, segments, length);
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        const auto tasks = count * slots;
        const auto groups = static_cast<unsigned>(tasks > 65535 * 256ULL ? 65535 : (tasks + 255) / 256);
        accumulate_segments<<<groups, 256, 0, queue>>>(view, reduction, static_cast<unsigned>(count), config, skip,
            segments, resident_segment_stride(static_cast<unsigned>(count)));
        error = cudaGetLastError();

        if (error != cudaSuccess) return static_cast<int>(error);
        count_segment_rays<<<blocks, 256, 0, queue>>>(view, segments, static_cast<unsigned>(count));
        error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    segment_phase<<<1, 1, 0, queue>>>(view, 8, 7);
    return static_cast<int>(cudaGetLastError());
}
}
