#include <covoxslam/sensor_grid_layout.hpp>
#include <algorithm>
#include <covoxslam/sensor_grid_precision.h>

namespace covoxslam::sensor_grid
{
operation_result<std::vector<std::uint16_t>> pack_ray_ids(std::span<const std::uint32_t> ids)
{
    if (std::any_of(ids.begin(), ids.end(), [](auto id) { return id > UINT16_MAX; }))
        return std::unexpected(operation_error{"sensor ray identifier exceeds sixteen bits"});
    std::vector<std::uint16_t> result(ids.size());

    for (std::size_t index = 0; index < ids.size(); ++index)
        result[index] = static_cast<std::uint16_t>(ids[index]);
    return result;
}
operation_result<hybrid_layout> build_hybrid_layout(const lookup& table, gather_backend backend,
    std::uint32_t threshold, std::size_t maximum_entries)
{
    if (!threshold || backend < gather_backend::hybrid_short_interleaved || backend > gather_backend::hybrid_ray16)
        return std::unexpected(operation_error{"invalid hybrid memory layout"});
    auto base = build_prepared_layout(table, false, maximum_entries);
    if (!base) return std::unexpected(base.error());
    auto classified = classify_rows(base->lengths, threshold);
    if (!classified) return std::unexpected(classified.error());
    hybrid_layout result{std::move(*base), std::move(*classified)};
    auto& data = result.data;
    auto& plan = result.plan;
    data.ranges.resize(data.cells.size());
    std::size_t entries{};
    const auto append = [&](std::size_t count) {
        if (entries > maximum_entries || count > maximum_entries - entries || count > UINT32_MAX - entries) return false;
        entries += count;
        return true;
    };
    if (backend == gather_backend::hybrid_short_interleaved)
    {
        data.short_stride = 32;

        for (std::size_t window = 0; window < plan.short_rows; window += 256) {
            const auto end = std::min<std::size_t>(window + 256, plan.short_rows);
            std::stable_sort(plan.rows.begin() + window, plan.rows.begin() + end,
                [&](auto a, auto b) { return data.lengths[a] < data.lengths[b]; });

            for (auto group = window; group < end; group += 32) {
                const auto group_end = std::min(group + 32, end);
                const auto start = entries;
                if (!append(std::size_t(data.lengths[plan.rows[group_end - 1]]) * 32))
                    return std::unexpected(operation_error{"hybrid padding exceeds entry budget"});

                for (auto worker = group; worker < group_end; ++worker) {
                    const auto row = plan.rows[worker];
                    data.ranges[row] = {static_cast<std::uint32_t>(start + worker - group), data.lengths[row]};
                }
            }
        }
    }

    for (std::size_t row = 0; row < data.cells.size(); ++row) {
        const auto count = data.lengths[row];
        if (data.short_stride == 32 && count <= threshold) continue;
        if (backend == gather_backend::hybrid_aligned && count > threshold && !append((8 - entries % 8) % 8))
            return std::unexpected(operation_error{"aligned padding exceeds entry budget"});
        const auto start = entries;
        if (!append(count)) return std::unexpected(operation_error{"hybrid rows exceed entry budget"});
        data.ranges[row] = {static_cast<std::uint32_t>(start), count};
    }
    std::vector<std::uint32_t> ids(entries);
    std::vector<float> entry_t(entries), projected(entries);

    for (std::size_t row = 0; row < data.cells.size(); ++row)
    {
        const auto range = data.ranges[row];
        const auto stride = range.count <= threshold ? data.short_stride : 1U;

        for (std::uint32_t step = 0; step < range.count; ++step) {
            const auto source = data.starts[row] + step;
            const auto destination = std::size_t(range.start) + std::size_t(step) * stride;
            ids[destination] = data.ray_ids[source];
            entry_t[destination] = data.entry_t[source];
            projected[destination] = data.projected[source];
        }
        data.starts[row] = range.start;
    }
    data.ray_ids = std::move(ids);
    data.entry_t = std::move(entry_t);
    data.projected = std::move(projected);
    if (backend == gather_backend::hybrid_ray16 && ray_ids_fit_u16(table.ray_count))
    {
        auto packed = pack_ray_ids(data.ray_ids);
        if (!packed) return std::unexpected(packed.error());
        data.ray_ids16 = std::move(*packed);
    }
     return result;
}
operation_result<subwarp_plan> classify_subwarps(std::span<const std::uint32_t> lengths)
{
    if (lengths.empty() || lengths.size() > 16777216 || std::find(lengths.begin(), lengths.end(), 0) != lengths.end())
        return std::unexpected(operation_error{"invalid subwarp row lengths"});
    subwarp_plan result;
    result.rows.reserve(lengths.size() * 2);

    for (unsigned policy = 0; policy < 2; ++policy)
    {
        const std::uint32_t limits[]{policy == 0 && accumulation_bits == 32 ? 16U : 32U, 64, 256, UINT32_MAX};
        std::uint32_t lower{};

        for (unsigned bin = 0; bin < 4; ++bin) {

            for (std::size_t row = 0; row < lengths.size(); ++row)
                if (lengths[row] > lower && lengths[row] <= limits[bin])
                    result.rows.push_back(static_cast<std::uint32_t>(row));
            lower = limits[bin];
            result.bin_ends[policy][bin] = static_cast<std::uint32_t>(result.rows.size());
        }
    }
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
