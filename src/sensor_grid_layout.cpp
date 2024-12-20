#include <covoxslam/sensor_grid_layout.hpp>
#include <algorithm>

namespace covoxslam::sensor_grid
{
operation_result<prepared_layout> build_prepared_layout(const lookup& table, bool interleaved,
    std::size_t maximum_entries)
{
    const auto valid = validate_lookup(table);
    if (!valid) return std::unexpected(valid.error());
    prepared_layout result;
    result.cells = table.active_voxels;
    result.stride = interleaved ? 32 : 1;
    const auto length = [&](std::uint32_t cell) { return table.offsets[cell + 1] - table.offsets[cell]; };

    if (interleaved) std::stable_sort(result.cells.begin(), result.cells.end(),
        [&](auto a, auto b) { return length(a) < length(b); });
    result.starts.resize(result.cells.size());
     result.lengths.resize(result.cells.size());
    std::size_t entries{};

    for (std::size_t base = 0; base < result.cells.size(); base += result.stride)
    {
        const auto width = std::min<std::size_t>(result.stride, result.cells.size() - base);
        std::uint32_t longest{};

        for (std::size_t lane = 0; lane < width; ++lane) longest = std::max(longest, length(result.cells[base + lane]));
        const std::size_t added = std::size_t(longest) * result.stride;
        if (entries > maximum_entries || added > maximum_entries - entries || added > UINT32_MAX - entries)
            return std::unexpected(operation_error{"prepared lookup exceeds entry budget"});

        for (std::size_t lane = 0; lane < width; ++lane) {
            result.starts[base + lane] = static_cast<std::uint32_t>(entries + lane);
            result.lengths[base + lane] = length(result.cells[base + lane]);
        }
        entries += added;
    }
    result.ray_ids.resize(entries);
    result.entry_t.resize(entries);
     result.projected.resize(entries);

    for (std::size_t row = 0; row < result.cells.size(); ++row)
    {
        const auto start = table.offsets[result.cells[row]];

        for (std::uint32_t step = 0; step < result.lengths[row]; ++step) {
            const auto& entry = table.intersections[start + step];
            const auto destination = std::size_t(result.starts[row]) + std::size_t(step) * result.stride;
            result.ray_ids[destination] = entry.ray;
             result.entry_t[destination] = entry.entry_t;
            result.projected[destination] = entry.projected;
        }
    }
    return result;
}
}
