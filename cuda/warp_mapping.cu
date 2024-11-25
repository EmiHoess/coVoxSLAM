#include "ray_resident_warp/integrate.cuh"
#include <initializer_list>
#include <cmath>
#include <limits>

namespace covoxslam::detail
{
int cuda_walk_resident_warp(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config,
    cuda_warp_parameters parameters, void* stream) noexcept
{
    if (!valid_resident_reduction_view(view, reduction, cuda_resident_reduction::direct_atomic) ||
        (count && !rays) || count > UINT32_MAX || !parameters.count || parameters.count > 65536)
        return static_cast<int>(cudaErrorInvalidValue);

    for (const float value : {config.voxel_size, config.truncation_distance,
        config.minimum_ray_length_m, config.maximum_ray_length_m})
        if (!std::isfinite(value) || value <= 0) return static_cast<int>(cudaErrorInvalidValue);

    if (config.minimum_ray_length_m > config.maximum_ray_length_m ||
        double(config.maximum_ray_length_m) + config.truncation_distance > std::numeric_limits<float>::max() ||
        (double(config.maximum_ray_length_m) + config.truncation_distance) / config.voxel_size >= 10000000)
        return static_cast<int>(cudaErrorInvalidValue);
    switch (parameters.cache_entries) {
    case 0: case 32: case 64: case 128: case 256: break;
     default: return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto queue = static_cast<cudaStream_t>(stream);
    begin_resident_warp<<<1, 1, 0, queue>>>(view);
    const auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    const auto blocks = (parameters.count + 3) / 4;
    const auto size = static_cast<unsigned>(count);
    switch (parameters.cache_entries)
    {
    case 0: walk_resident_warp<0><<<blocks, 128, 0, queue>>>(view, reduction, rays, size, config, parameters.scattered); break;
    case 32: walk_resident_warp<32><<<blocks, 128, 0, queue>>>(view, reduction, rays, size, config, parameters.scattered); break;
    case 64: walk_resident_warp<64><<<blocks, 128, 0, queue>>>(view, reduction, rays, size, config, parameters.scattered); break;
    case 128: walk_resident_warp<128><<<blocks, 128, 0, queue>>>(view, reduction, rays, size, config, parameters.scattered); break;
    case 256: walk_resident_warp<256><<<blocks, 128, 0, queue>>>(view, reduction, rays, size, config, parameters.scattered); break;
    }
    return static_cast<int>(cudaGetLastError());
}
}
