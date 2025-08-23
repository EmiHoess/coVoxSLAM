#include <covoxslam/sensor/prepared_layout.hpp>
#include <algorithm>
#include <covoxslam/sensor_grid_precision.h>

namespace covoxslam::sensor_grid
{
operation_status validate_row_plan(std::span<const std::uint32_t> lengths, const row_plan& plan,
    std::uint32_t short_max, std::uint32_t block_min)
{
    if (lengths.empty() || lengths.size() > 16777216 || plan.rows.size() != lengths.size() ||
        plan.short_rows > plan.rows.size() || plan.block_rows > plan.rows.size() - plan.short_rows ||
        !short_max || block_min <= short_max)
        return std::unexpected(operation_error{"invalid row partition dimensions"});
    std::vector<bool> visited(lengths.size());
    const auto block_start = plan.rows.size() - plan.block_rows;

    for (std::size_t worker = 0; worker < plan.rows.size(); ++worker) {
        const auto row = plan.rows[worker];
        if (row >= lengths.size() || visited[row] || !lengths[row])
            return std::unexpected(operation_error{"row partition does not have unique ownership"});
        visited[row] = true;
        const auto length = lengths[row];
        const bool eligible = worker < plan.short_rows ? length <= short_max :
            worker < block_start ? length > short_max && length < block_min : length >= block_min;
        if (!eligible) return std::unexpected(operation_error{"row partition threshold mismatch"});
    }
    return {};
}
operation_result<row_plan> classify_blocks(std::span<const std::uint32_t> lengths,
    std::uint32_t short_max, std::uint32_t block_min)
{
    if (!short_max || block_min <= short_max)
        return std::unexpected(operation_error{"block threshold must exceed short row threshold"});
    auto result = classify_rows(lengths, short_max);
    if (!result) return std::unexpected(result.error());
    const auto split = std::stable_partition(result->rows.begin() + result->short_rows, result->rows.end(),
        [&](auto row) { return lengths[row] < block_min; });
    result->block_rows = static_cast<std::uint32_t>(result->rows.end() - split);
    const auto valid = validate_row_plan(lengths, *result, short_max, block_min);
    if (!valid) return std::unexpected(valid.error());
     return result;
}
operation_result<std::vector<std::uint16_t>> pack_ray_ids(std::span<const std::uint32_t> ids)
{
    if (std::any_of(ids.begin(), ids.end(), [](auto id) { return id > UINT16_MAX; }))
        return std::unexpected(operation_error{"sensor ray identifier exceeds sixteen bits"});
    std::vector<std::uint16_t> result(ids.size());

    for (std::size_t index = 0; index < ids.size(); ++index)
        result[index] = static_cast<std::uint16_t>(ids[index]);
    return result;
}
operation_result<row_plan> classify_rows(std::span<const std::uint32_t> lengths, std::uint32_t threshold)
{
    if (lengths.empty() || lengths.size() > UINT32_MAX || std::find(lengths.begin(), lengths.end(), 0) != lengths.end())
        return std::unexpected(operation_error{"invalid row classification lengths"});
    row_plan result;
    result.rows.reserve(lengths.size());

    for (std::size_t row = 0; row < lengths.size(); ++row)
        if (lengths[row] <= threshold) result.rows.push_back(static_cast<std::uint32_t>(row));
    result.short_rows = static_cast<std::uint32_t>(result.rows.size());

    for (std::size_t row = 0; row < lengths.size(); ++row)
        if (lengths[row] > threshold) result.rows.push_back(static_cast<std::uint32_t>(row));
    return result;
}
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
