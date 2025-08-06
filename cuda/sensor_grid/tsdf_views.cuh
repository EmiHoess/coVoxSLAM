#pragma once
#include <covoxslam/storage/device_view.hpp>
#include "../common/mapping_types.h"

namespace covoxslam::sensor_grid
{
using sensor_tsdf_view = storage::device_map_view<storage::schema::tsdf>;
}
