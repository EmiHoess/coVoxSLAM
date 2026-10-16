#include <covoxslam/sensor/lookup.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace covoxslam::sensor_grid
{
namespace
{
using unit_direction = std::array<double, 3>;
inline constexpr std::uint64_t voxel_storage_allowance_bytes = 24;
inline constexpr std::uint64_t offset_terminal_allowance_bytes = 8;
inline constexpr std::uint64_t ray_storage_allowance_bytes = 32;

operation_result<std::uint64_t> lookup_peak_bytes(
    std::uint64_t voxel_count, std::uint64_t ray_count, std::uint64_t intersection_count)
{
    std::uint64_t result = offset_terminal_allowance_bytes;
    const auto append = [&result](std::uint64_t count, std::uint64_t width) {
        if (count > (std::numeric_limits<std::uint64_t>::max() - result) / width)
            return false;
        result += count * width;
        return true;
    };
    if (!append(voxel_count, voxel_storage_allowance_bytes) ||
        !append(ray_count, ray_storage_allowance_bytes) ||
        !append(intersection_count, sizeof(intersection)))
        return std::unexpected(operation_error{"sensor lookup byte estimate overflows"});
    return result;
}

inline operation_result<unit_direction> normalize(direction input)
{
    const double norm = std::hypot(double(input.x), double(input.y), double(input.z));

    if (!std::isfinite(norm) || norm <= 0)
        return std::unexpected(operation_error{"sensor ray must be finite and nonzero"});
    return unit_direction{input.x / norm, input.y / norm, input.z / norm};
}
template<class emit_type>
operation_status walk(const geometry& grid, const config& c,
                      const unit_direction& ray, emit_type&& emit)
{
    std::array<int, 3> cell{}, step{};
     std::array<double, 3> next{};
    const auto crossing = [&](unsigned axis) {
        return step[axis] == 0 ? std::numeric_limits<double>::infinity() :
            double(cell[axis] + (step[axis] > 0)) * c.voxel_size / ray[axis];
    };

    for (unsigned axis = 0; axis < 3; ++axis)
    {
        step[axis] = ray[axis] > 0 ? 1 : ray[axis] < 0 ? -1 : 0;
        cell[axis] = step[axis] < 0 ? -1 : 0;
        next[axis] = crossing(axis);
    }
    double entry = 0;
    const int half = static_cast<int>(grid.half_side);

    while (entry < c.radius)
    {
        std::array<std::uint32_t, 3> index{};
        double projected = 0;

        for (unsigned axis = 0; axis < 3; ++axis) {
            if (cell[axis] < -half || cell[axis] >= half) return {};
             index[axis] = static_cast<std::uint32_t>(cell[axis] + half);
            projected += (cell[axis] + 0.5) * c.voxel_size * ray[axis];
        }
        const auto voxel = index[0] + grid.side * (index[1] + grid.side * index[2]);

        if (!std::isfinite(static_cast<float>(projected)))
            return std::unexpected(operation_error{"sensor projection exceeds capacity"});
        const auto status = emit(voxel, static_cast<float>(entry), static_cast<float>(projected));
        if (!status) return status;
        const double distance = std::min({next[0], next[1], next[2]});

        if (distance >= c.radius) break;
        const double tolerance = 8 * std::numeric_limits<double>::epsilon() * distance;

        for (unsigned axis = 0; axis < 3; ++axis)
            if (next[axis] <= distance + tolerance)
            {
                cell[axis] += step[axis];
                 next[axis] = crossing(axis);
            }
        entry = distance;
    }
    return {};
}
}

operation_result<geometry> make_geometry(const config& c)
{
    if (!std::isfinite(c.voxel_size) || c.voxel_size <= 0 ||
        !std::isfinite(c.radius) || c.radius <= 0 ||
        !std::isfinite(c.truncation) || c.truncation <= 0 ||
        !std::isfinite(c.minimum_range) || c.minimum_range < 0 ||
        !std::isfinite(c.maximum_weight) || c.maximum_weight <= 0 ||
        !std::isfinite(c.sparsity_compensation) || c.sparsity_compensation <= 0 ||
        (c.weight_dropoff && c.truncation <= c.voxel_size))
        return std::unexpected(operation_error{"invalid sensor grid configuration"});
    const double half = std::ceil(double(c.radius) / c.voxel_size);
    constexpr auto capacity = std::numeric_limits<std::uint32_t>::max();

    if (half < 1 || half > capacity / 2U)
        return std::unexpected(operation_error{"sensor grid side exceeds capacity"});
     const auto side = static_cast<std::uint64_t>(half) * 2U;
    if (!dense_grid_volume_fits(static_cast<std::uint32_t>(side)))
        return std::unexpected(operation_error{"sensor grid volume exceeds capacity"});
    const double extent = half * c.voxel_size;

    if (!std::isfinite(extent) || extent > std::numeric_limits<float>::max())
        return std::unexpected(operation_error{"sensor grid origin exceeds capacity"});
    return geometry{static_cast<std::uint32_t>(half), static_cast<std::uint32_t>(side),
        static_cast<std::uint32_t>(side * side * side), static_cast<float>(-extent)};
}
operation_result<lookup> build_lookup(const config& c, std::span<const direction> directions,
                                      std::uint32_t factor, lookup_limits limits)
{
    const auto grid = make_geometry(c);
    if (!grid) return std::unexpected(grid.error());

    if (directions.empty() || directions.size() > UINT32_MAX || factor == 0)
        return std::unexpected(operation_error{"sensor lookup requires rays and a positive factor"});
    const auto base_bytes = lookup_peak_bytes(grid->voxels, directions.size(), 0);

    if (grid->voxels > limits.maximum_voxels || directions.size() > limits.maximum_rays ||
        !base_bytes || *base_bytes > limits.maximum_host_bytes || !limits.maximum_intersections)
        return std::unexpected(operation_error{"sensor lookup exceeds host storage budget"});
    std::vector<unit_direction> normalized;
    normalized.reserve(directions.size());

    for (const auto input : directions) {
        const auto ray = normalize(input);
        if (!ray) return std::unexpected(ray.error());
         normalized.push_back(*ray);
    }
    lookup table;
    table.settings = c;
    table.grid = *grid;
    table.ray_count = static_cast<std::uint32_t>(directions.size());
    table.reduction_factor = factor;
    table.selected_rays = (table.ray_count - 1) / factor + 1;
    table.offsets.assign(std::size_t(grid->voxels) + 1, 0);
    std::uint64_t total{};

    for (std::size_t ray = 0; ray < normalized.size(); ray += factor)
    {
        const auto status = walk(*grid, c, normalized[ray],
            [&](std::uint32_t voxel, float, float) -> operation_status {
                if (++total > UINT32_MAX)
                    return std::unexpected(operation_error{"sensor intersections exceed CSR capacity"});
                const auto peak_bytes = lookup_peak_bytes(grid->voxels, directions.size(), total);
                if (total > limits.maximum_intersections || !peak_bytes ||
                    *peak_bytes > limits.maximum_host_bytes)
                    return std::unexpected(operation_error{"sensor intersections exceed host storage budget"});
                 ++table.offsets[std::size_t(voxel) + 1];
                return {};
            });
        if (!status) return std::unexpected(status.error());
    }

    for (std::size_t voxel = 0; voxel < grid->voxels; ++voxel) {
        if (table.offsets[voxel + 1]) table.active_voxels.push_back(static_cast<std::uint32_t>(voxel));
        table.offsets[voxel + 1] += table.offsets[voxel];
    }
    table.intersections.resize(static_cast<std::size_t>(total));
    auto cursor = table.offsets;

    for (std::size_t ray = 0; ray < normalized.size(); ray += factor)
    {
        const auto status = walk(*grid, c, normalized[ray],
            [&](std::uint32_t voxel, float entry, float projected) -> operation_status {
                if (cursor[voxel] >= table.offsets[std::size_t(voxel) + 1])
                    return std::unexpected(operation_error{"sensor row count changed during construction"});
                table.intersections[cursor[voxel]++] = {static_cast<std::uint32_t>(ray), entry, projected};
                return {};
            });
        if (!status) return std::unexpected(status.error());
    }
    const auto status = validate_lookup(table);
    if (!status) return std::unexpected(status.error());
    return table;
}

operation_status validate_lookup(const lookup& table)
{
    const auto grid = make_geometry(table.settings);
    if (!grid) return std::unexpected(grid.error());

    if (table.grid.side != grid->side || table.grid.half_side != grid->half_side ||
        table.grid.voxels != grid->voxels || table.grid.origin != grid->origin ||
        table.offsets.size() != std::size_t(grid->voxels) + 1 || !table.ray_count ||
        !table.reduction_factor || table.selected_rays != (table.ray_count - 1) / table.reduction_factor + 1 ||
        table.intersections.size() > UINT32_MAX || table.intersections.size() < table.selected_rays ||
        table.offsets.front() != 0 || table.offsets.back() != table.intersections.size())
        return std::unexpected(operation_error{"malformed sensor lookup dimensions"});
    std::vector<bool> seen(table.selected_rays, false);
    std::size_t active{};

    for (std::size_t voxel = 0; voxel < grid->voxels; ++voxel)
    {
        const auto begin = table.offsets[voxel], end = table.offsets[voxel + 1];

        if (begin > end || end > table.intersections.size())
            return std::unexpected(operation_error{"malformed sensor row offsets"});
        if (begin != end && (active >= table.active_voxels.size() || table.active_voxels[active++] != voxel))
            return std::unexpected(operation_error{"malformed sensor active cells"});
        std::uint32_t previous{};

        for (auto index = begin; index < end; ++index) {
            const auto& hit = table.intersections[index];

            if (hit.ray >= table.ray_count || hit.ray % table.reduction_factor ||
                (index != begin && hit.ray <= previous) || !std::isfinite(hit.entry_t) ||
                hit.entry_t < 0 || hit.entry_t > table.settings.radius ||
                !std::isfinite(hit.projected) || hit.projected < 0)
                return std::unexpected(operation_error{"malformed sensor intersection"});
             previous = hit.ray;
            seen[hit.ray / table.reduction_factor] = true;
        }
    }

    if (active != table.active_voxels.size() || std::find(seen.begin(), seen.end(), false) != seen.end())
        return std::unexpected(operation_error{"sensor lookup has missing rays or extra active cells"});
    return {};
}
}
