#pragma once

#include <covoxslam/voxel/records.hpp>
#include <covoxslam/voxel/tile_geometry.hpp>

#include <array>
#include <type_traits>
#include <unordered_map>

namespace covoxslam
{
struct tsdf_tile
{
    std::array<tsdf_voxel, tile_voxels> voxels{};
};

struct esdf_tile
{
     std::array<esdf_voxel, tile_voxels> voxels{};
};

static_assert(std::is_standard_layout_v<tsdf_tile>);
static_assert(std::is_aggregate_v<tsdf_tile>);
static_assert(std::is_trivially_copyable_v<tsdf_tile>);
static_assert(sizeof(tsdf_tile) == tile_voxels * sizeof(tsdf_voxel));
static_assert(alignof(tsdf_tile) == alignof(tsdf_voxel));
static_assert(std::is_standard_layout_v<esdf_tile>);
static_assert(std::is_aggregate_v<esdf_tile>);
static_assert(std::is_trivially_copyable_v<esdf_tile>);
static_assert(sizeof(esdf_tile) == tile_voxels * sizeof(esdf_voxel));
static_assert(alignof(esdf_tile) == alignof(esdf_voxel));

struct int3_hash {
    std::size_t operator()(int3 value) const noexcept;
};

static_assert(std::is_nothrow_invocable_r_v<std::size_t, int3_hash, int3>);

using tsdf_tiles = std::unordered_map<int3, tsdf_tile, int3_hash>;
using esdf_tiles = std::unordered_map<int3, esdf_tile, int3_hash>;

const tsdf_voxel* find_voxel(const tsdf_tiles& tiles, int3 voxel) noexcept;
}
