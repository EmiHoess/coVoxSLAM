#pragma once

#include <cstddef>

namespace covoxslam
{
inline constexpr int tile_edge = 8;
inline constexpr std::size_t tile_face_voxels =
    static_cast<std::size_t>(tile_edge) * static_cast<std::size_t>(tile_edge);
inline constexpr std::size_t tile_voxels = tile_face_voxels * static_cast<std::size_t>(tile_edge);

static_assert(tile_edge == 8);
static_assert(tile_face_voxels == 64);
static_assert(tile_voxels == 512);
}
