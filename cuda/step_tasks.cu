#include "ray_resident_step_tasks/integrate.cuh"
#include <algorithm>
#include <cmath>

namespace covoxslam::detail
{
int cuda_process_resident_tasks(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_resident_task_view tasks, const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters config, unsigned stage, void* stream) noexcept
{
    if (!valid_resident_reduction_view(view, reduction, cuda_resident_reduction::direct_atomic) ||
        !valid_resident_tasks(tasks) || count > tasks.ray_capacity || (count && !rays) || stage > 1 ||
        !std::isfinite(config.voxel_size) || config.voxel_size <= 0 ||
        !std::isfinite(config.truncation_distance) || config.truncation_distance <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    const auto queue = static_cast<cudaStream_t>(stream);
    const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(65535, (tasks.capacity + 255) / 256));
    const auto phase = stage + 3;
    check_resident_task_phase<<<1, 1, 0, queue>>>(view, tasks, phase, static_cast<std::uint32_t>(count), false);
    auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    if (!stage) evaluate_resident_tasks<<<blocks, 256, 0, queue>>>(view, tasks, rays, config);
    else accumulate_resident_tasks<<<blocks, 256, 0, queue>>>(view, reduction, tasks, config.truncation_distance);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);

    if (stage && count) {
        count_resident_task_rays<<<static_cast<unsigned>((count + 255) / 256), 256, 0, queue>>>(view, tasks);
        error = cudaGetLastError();
    }
    if (error != cudaSuccess) return static_cast<int>(error);
    check_resident_task_phase<<<1, 1, 0, queue>>>(view, tasks, phase, static_cast<std::uint32_t>(count), true);
    return static_cast<int>(cudaGetLastError());
}
}
