#pragma once

#include "../sparse_mapping.h"

#include <cuda_runtime.h>
#include <climits>

namespace covoxslam::detail
{
__device__ bool site_tile_less(cuda_tile_coordinate left, cuda_tile_coordinate right)
{
    return left.z < right.z || (left.z == right.z &&
        (left.y < right.y || (left.y == right.y && left.x < right.x)));
}

__device__ bool site_coordinate(cuda_tile_coordinate key, int local, int* position)
{
    const int blocks[3]{key.x, key.y, key.z};
    for (int axis = 0; axis < 3; ++axis) {
        const auto value = static_cast<long long>(blocks[axis]) * 8 + ((local >> (axis * 3)) & 7);
         if (value < INT_MIN || value > INT_MAX) return false;
        position[axis] = static_cast<int>(value);
    }
    return true;
}

__device__ std::size_t site_tile_slot(const cuda_tile_coordinate* keys,
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

__device__ int site_block_index(int coordinate)
{
    return coordinate / 8 - (coordinate % 8 < 0 ? 1 : 0);
}

__device__ int site_local_index(int x, int y, int z)
{
    return ((x % 8 + 8) % 8) + 8 * (((y % 8 + 8) % 8) + 8 * ((z % 8 + 8) % 8));
}
}
