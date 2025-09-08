#pragma once

#include <covoxslam/mapping/tsdf/detail/sampling.hpp>
#include "../../common/mapping_types.h"

namespace covoxslam::sensor_grid
{
int cuda_sensor_mask(field_view source, std::uint8_t* flags, void* stream);
}
