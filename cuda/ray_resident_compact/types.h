#pragma once

#include "../ray_resident_reduce/types.h"

namespace covoxslam::detail
{
struct cuda_compact_view { unsigned* touched{}; };

int cuda_resident_compact_stage(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_fusion_parameters fusion,
    cuda_free_space_view skip, cuda_compact_view compact, unsigned stage, void* stream, bool batched = false) noexcept;
}
