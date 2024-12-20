#pragma once
#include <covoxslam/sensor_grid.hpp>
#include <covoxslam/sensor_grid_layout_types.h>

namespace covoxslam::sensor_grid
{
struct prepared_layout
{
    std::vector<std::uint32_t> starts, lengths, cells, ray_ids;
    std::vector<float> entry_t, projected;
    std::uint32_t stride{1};
    layout_view view() const noexcept
    {
        return {starts.data(), lengths.data(), cells.data(), ray_ids.data(), entry_t.data(), projected.data(),
            static_cast<std::uint32_t>(cells.size()), static_cast<std::uint32_t>(ray_ids.size()), stride};
    }
};
operation_result<prepared_layout> build_prepared_layout(const lookup& table, bool interleaved,
    std::size_t maximum_entries = 67108864);
}
