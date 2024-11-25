#pragma once

#include "../ray_resident_reduce/types.h"

namespace covoxslam::detail
{
struct cuda_warp_parameters
{
    unsigned count{256}, cache_entries{};
    bool scattered{};
};

int cuda_walk_resident_warp(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config,
    cuda_warp_parameters parameters, void* stream) noexcept;
}
