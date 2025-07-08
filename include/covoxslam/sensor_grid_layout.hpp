#pragma once
#include <covoxslam/sensor_grid.hpp>
#include <covoxslam/sensor_grid_layout_types.h>

namespace covoxslam::sensor_grid
{
struct prepared_layout
{
    std::vector<std::uint32_t> starts, lengths, cells, ray_ids;
    std::vector<float> entry_t, projected;
    std::vector<row_range> ranges;
    std::vector<std::uint16_t> ray_ids16;
    std::uint32_t stride{1};
    std::uint32_t short_stride{1};
    layout_view view() const noexcept
    {
        return {starts.data(), lengths.data(), cells.data(), ray_ids.data(), entry_t.data(), projected.data(),
            static_cast<std::uint32_t>(cells.size()), static_cast<std::uint32_t>(ray_ids.size()), stride};
    }
};
operation_result<prepared_layout> build_prepared_layout(const lookup& table, bool interleaved,
    std::size_t maximum_entries = 67108864);
struct row_plan { std::vector<std::uint32_t> rows; std::uint32_t short_rows{}, block_rows{}; };
operation_result<row_plan> classify_rows(std::span<const std::uint32_t> lengths, std::uint32_t threshold);
operation_result<row_plan> classify_blocks(std::span<const std::uint32_t> lengths,
    std::uint32_t short_max, std::uint32_t block_min);
operation_status validate_row_plan(std::span<const std::uint32_t> lengths, const row_plan& plan,
    std::uint32_t short_max, std::uint32_t block_min);
operation_result<std::vector<std::uint16_t>> pack_ray_ids(std::span<const std::uint32_t> ids);
constexpr bool ray_ids_fit_u16(std::uint32_t original_ray_count) noexcept
{
    return original_ray_count > 0 && original_ray_count <= 65536;
}
}
