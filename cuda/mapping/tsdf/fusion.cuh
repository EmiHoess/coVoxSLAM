#pragma once

#include <covoxslam/mapping/tsdf/detail/sampling.hpp>
#include "../detail/types.cuh"

namespace covoxslam::sensor_grid
{
int cuda_sensor_mask(
    field_view source, std::uint8_t* flags, std::uint64_t& kernel_launches, void* stream);
}
