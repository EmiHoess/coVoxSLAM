#pragma once

#include <covoxslam/mapping.hpp>
#include "sparse_mapping.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace covoxslam::detail
{
inline esdf_tiles convert_cuda_site_esdf(std::span<const cuda_tile_coordinate> keys,
    std::span<const cuda_site_esdf_voxel> values, cuda_site_esdf_parameters parameters)
{
    const float maximum_voxels = parameters.maximum_distance_m / parameters.voxel_size;
    const float maximum_squared = maximum_voxels * maximum_voxels;
    if (!std::isfinite(parameters.voxel_size) || parameters.voxel_size <= 0 ||
        !std::isfinite(parameters.maximum_distance_m) || parameters.maximum_distance_m <= 0 ||
        !std::isfinite(maximum_squared) || maximum_squared <= 0 ||
        keys.size() > std::numeric_limits<std::size_t>::max() / tile_voxels ||
        values.size() != keys.size() * tile_voxels)
        throw std::invalid_argument("invalid CUDA ESDF conversion shape or scale");
    esdf_tiles result;
    result.reserve(keys.size());
    for (std::size_t slot = 0; slot < keys.size(); ++slot)
    {
        const int3 block{keys[slot].x, keys[slot].y, keys[slot].z};
        if (slot && !block_index_less({keys[slot - 1].x, keys[slot - 1].y, keys[slot - 1].z}, block))
            throw std::invalid_argument("CUDA ESDF tile keys must be strictly ordered");
         esdf_tile tile;
        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto position = global_index(block, local);
            const auto& source = values[slot * tile_voxels + local];
            const auto flags = source.flags;
            const bool observed = (flags & cuda_esdf_observed) != 0;
            const bool assigned = (flags & cuda_esdf_parent_assigned) != 0;
            const bool site = (flags & cuda_esdf_site) != 0;
            const float squared = source.squared_distance_voxels;
            if ((flags & ~15U) || !std::isfinite(squared) || squared < 0 ||
                (!observed && (flags & ~cuda_esdf_observed)) ||
                (site && (!assigned || !(flags & cuda_esdf_inside) || squared != 0)) ||
                (!assigned && squared < maximum_squared))
                throw std::invalid_argument("invalid CUDA ESDF distance or flags");
            const bool clipped = squared >= maximum_squared;
            int parents[3]{};
            if (assigned && !clipped)
            {
                const int coordinates[3]{position.x, position.y, position.z};
                for (int axis = 0; axis < 3; ++axis) {
                    const auto offset = std::int64_t(source.parent[axis]) - coordinates[axis];
                     if (offset < std::numeric_limits<int>::min() || offset > std::numeric_limits<int>::max())
                        throw std::out_of_range("CUDA ESDF relative parent exceeds integer range");
                    parents[axis] = static_cast<int>(offset);
                }
                if (site && (parents[0] || parents[1] || parents[2]))
                    throw std::invalid_argument("CUDA ESDF site parent is not itself");
            }
            const float distance = clipped ? parameters.maximum_distance_m :
                std::min(parameters.maximum_distance_m, parameters.voxel_size * std::sqrt(squared));
            tile.voxels[local] = {std::copysign(distance, (flags & cuda_esdf_inside) ? -1.0F : 1.0F),
                parents[0], parents[1], parents[2], flags & (observed_flag | esdf_site_flag)};
        }
        result.emplace(block, std::move(tile));
    }
    return result;
}
}
