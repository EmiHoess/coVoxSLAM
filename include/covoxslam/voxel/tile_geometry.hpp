#pragma once

#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/voxel/tile_shape.hpp>

#include <cstddef>

namespace covoxslam
{
static_assert(sizeof(int) == 4);

int3 block_index(int3 voxel) noexcept;
operation_result<int3> global_index(int3 block, std::size_t local);
vec3f voxel_center(int3 index, float size) noexcept;
std::size_t local_index(int3 voxel) noexcept;
bool block_index_less(int3 left, int3 right) noexcept;
}
