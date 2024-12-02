#pragma once

#include "../ray_resident/types.h"
#include "../ray_resident_free_space_skip/types.h"

namespace covoxslam::detail
{
struct cuda_fp32_view { float* weights{}; float* weighted_distances{}; };

int cuda_resident_fp32_stage(cuda_resident_view view, cuda_fp32_view sums,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_fusion_parameters fusion,
    cuda_free_space_view skip, unsigned stage, void* stream) noexcept;
}
