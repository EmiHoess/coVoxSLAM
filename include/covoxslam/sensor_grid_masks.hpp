#pragma once
#include <covoxslam/sensor_grid.hpp>
#include <covoxslam/sensor_grid_fp32.h>

namespace covoxslam::sensor_grid
{
enum class source_mask_mode : std::uint32_t { dense, compact };
enum class candidate_mode : std::uint32_t { whole_tiles, voxel_mask, compacted_control, compact_fp32 };
struct transfer_options
{
    source_mask_mode source_mask{source_mask_mode::dense};
    transfer_arithmetic arithmetic{transfer_arithmetic::fp64};
    bool specialized{};
    candidate_mode candidates{candidate_mode::whole_tiles};
    std::uint32_t candidate_capacity{1048576};
};
struct compact_mask_layout
{
    std::vector<std::uint32_t> offsets, cells;
};
inline operation_result<compact_mask_layout> build_compact_mask(const lookup& table)
{
    const auto status = validate_lookup(table);

    if (!status) return std::unexpected(status.error());
    const auto side = table.grid.side;
    const auto n = (side + 7) / 8;
    const std::size_t tiles = std::size_t(n) * n * n;
    compact_mask_layout result;
    result.offsets.resize(tiles + 1);
    result.cells.resize(table.active_voxels.size());
    const auto tile_of = [side, n](std::uint32_t cell) {
        const auto x = cell % side, y = cell / side % side, z = cell / side / side;
        return x / 8 + std::size_t(n) * (y / 8 + std::size_t(n) * (z / 8));
    };

    for (auto cell : table.active_voxels) ++result.offsets[tile_of(cell) + 1];

    for (std::size_t tile = 0; tile < tiles; ++tile)
        result.offsets[tile + 1] += result.offsets[tile];
    auto cursor = result.offsets;

    for (auto cell : table.active_voxels)
        result.cells[cursor[tile_of(cell)]++] = cell;
    if (result.offsets.back() != result.cells.size())
        return std::unexpected(operation_error{"invalid compact mask coverage"});

    for (std::size_t tile = 0; tile < tiles; ++tile)
        if (cursor[tile] != result.offsets[tile + 1] || result.offsets[tile + 1] - result.offsets[tile] > 512)
            return std::unexpected(operation_error{"invalid compact mask tile ownership"});
    return result;
}
}
