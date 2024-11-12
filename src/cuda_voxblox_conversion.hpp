#pragma once

#include <covoxslam/mapping.hpp>
#include "sparse_mapping.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace covoxslam::detail
{
inline esdf_tiles convert_cuda_voxblox(std::span<const cuda_tile_coordinate> keys,
    std::span<const cuda_tsdf_tile> input, std::span<const cuda_voxblox_voxel> values, cuda_blocks_parameters config)
{
    if (keys.size() > UINT32_MAX / tile_voxels || input.size() != keys.size() ||
        values.size() != keys.size() * tile_voxels) throw std::invalid_argument("invalid Voxblox conversion shape");
    esdf_tiles output;
    for (std::size_t slot = 0; slot < keys.size(); ++slot)
    {
        const int3 block{keys[slot].x, keys[slot].y, keys[slot].z};
        if (slot && !block_index_less({keys[slot - 1].x, keys[slot - 1].y, keys[slot - 1].z}, block))
            throw std::invalid_argument("Voxblox keys must be strictly ordered");
        esdf_tile tile;
        bool active{};
        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto position = global_index(block, local);
            const auto& source = input[slot].voxels[local];
            const auto& value = values[slot * tile_voxels + local];
             const bool observed = (source.flags & observed_flag) && source.weight >= config.minimum_weight;
            const bool seed = observed && std::abs(source.distance) < config.seed_band_m;
            const auto flags = observed ? 1U | (source.distance <= 0 ? 2U : 0U) | (seed ? 4U : 0U) : 0U;
            if (value.flags != flags || !std::isfinite(value.distance_m) ||
                (seed && (std::memcmp(&source.distance, &value.distance_m, sizeof(float)) != 0 ||
                    value.parent[0] || value.parent[1] || value.parent[2])) ||
                (!observed && (value.distance_m || value.parent[0] || value.parent[1] || value.parent[2])))
                throw std::invalid_argument("invalid Voxblox seed or observed domain");
            const int coordinates[3]{position.x, position.y, position.z};
            for (int axis = 0; axis < 3; ++axis) {
                const auto parent = std::int64_t(coordinates[axis]) + value.parent[axis];
                if (parent < INT32_MIN || parent > INT32_MAX) throw std::out_of_range("Voxblox parent exceeds coordinate domain");
            }
            const bool clipped = std::abs(value.distance_m) >= config.maximum_distance_m;
            const float distance = std::copysign(std::min(std::abs(value.distance_m), config.maximum_distance_m), value.distance_m);
            tile.voxels[local] = {distance, clipped ? 0 : value.parent[0], clipped ? 0 : value.parent[1],
                clipped ? 0 : value.parent[2], value.flags & 5U};
            active = active || observed;
        }
         if (active) output.emplace(block, std::move(tile));
    }
    return output;
}
}
