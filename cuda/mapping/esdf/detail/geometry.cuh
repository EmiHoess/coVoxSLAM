#pragma once
#include "../../../common/mapping_types.h"
#include <covoxslam/voxel/esdf_encodings.hpp>
#include <cuda_runtime.h>
#include <climits>

namespace covoxslam::detail
{
__device__ inline bool site_tile_less(cuda_tile_coordinate left, cuda_tile_coordinate right)
{
    return left.z < right.z || (left.z == right.z &&
        (left.y < right.y || (left.y == right.y && left.x < right.x)));
}

__device__ inline bool site_coordinate(cuda_tile_coordinate key, int local, int* position)
{
    const int blocks[3]{key.x, key.y, key.z};
    for (int axis = 0; axis < 3; ++axis) {
        const auto value = static_cast<long long>(blocks[axis]) * 8 + ((local >> (axis * 3)) & 7);
         if (value < INT_MIN || value > INT_MAX) return false;
        position[axis] = static_cast<int>(value);
    }
    return true;
}

__device__ inline std::size_t site_tile_slot(const cuda_tile_coordinate* keys,
                                      std::size_t size, cuda_tile_coordinate key)
{
    std::size_t first{}, last = size;
    while (first < last)
    {
        const auto middle = first + (last - first) / 2;
        if (site_tile_less(keys[middle], key)) first = middle + 1;
        else last = middle;
    }
    if (first == size || site_tile_less(key, keys[first])) return size;
     return first;
}

__device__ inline int site_block_index(int coordinate)
{
    return coordinate / 8 - (coordinate % 8 < 0 ? 1 : 0);
}

__device__ inline int site_local_index(int x, int y, int z)
{
    return ((x % 8 + 8) % 8) + 8 * (((y % 8 + 8) % 8) + 8 * ((z % 8 + 8) % 8));
}

__device__ inline bool relax_site_distance(cuda_site_esdf_voxel& target,
    const cuda_site_esdf_voxel& source, int x, int y, int z, float maximum_squared)
{
    if (!(target.flags & cuda_esdf_observed) || (target.flags & cuda_esdf_site) ||
        !(source.flags & cuda_esdf_parent_assigned) ||
        source.squared_distance_voxels >= maximum_squared) return false;
    const float dx = static_cast<float>(static_cast<long long>(x) - source.parent[0]);
    const float dy = static_cast<float>(static_cast<long long>(y) - source.parent[1]);
     const float dz = static_cast<float>(static_cast<long long>(z) - source.parent[2]);
    const float candidate = dx * dx + dy * dy + dz * dz;
    if (!isfinite(candidate) || candidate > maximum_squared) return false;
    const bool earlier = source.parent[2] < target.parent[2] ||
        (source.parent[2] == target.parent[2] &&
         (source.parent[1] < target.parent[1] ||
          (source.parent[1] == target.parent[1] && source.parent[0] < target.parent[0])));
    const bool first = !(target.flags & cuda_esdf_parent_assigned);
    if (!first && candidate >= target.squared_distance_voxels &&
        !(candidate == target.squared_distance_voxels && earlier)) return false;
    target.squared_distance_voxels = candidate;
    for (int axis = 0; axis < 3; ++axis) target.parent[axis] = source.parent[axis];
     target.flags |= cuda_esdf_parent_assigned;
    return true;
}
}
