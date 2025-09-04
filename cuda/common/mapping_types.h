#pragma once
#include <covoxslam/voxel/records.hpp>
#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
struct cuda_tile_coordinate
{
    std::int32_t x{}, y{}, z{};
};

using cuda_tsdf_voxel = tsdf_record;

struct cuda_projective_pose
{
    float translation[3]{};
    float rotation_xyzw[4]{0, 0, 0, 1};
};
}
