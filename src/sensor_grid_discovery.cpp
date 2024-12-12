#include <covoxslam/sensor_grid_map.hpp>
#include <algorithm>
#include <cmath>
#include <climits>
#include <set>

namespace covoxslam::sensor_grid
{
operation_result<std::vector<int3>> destination_tiles(geometry grid, float size,
    std::span<const std::uint8_t> observed, pose transform, std::size_t capacity)
{
    const auto q = transform.rotation;
    const double norm = double(q.x) * q.x + double(q.y) * q.y + double(q.z) * q.z + double(q.w) * q.w;
    if (!grid.side || grid.side > 1624 || !capacity || !std::isfinite(size) || size <= 0 ||
        !std::isfinite(grid.origin) || !std::isfinite(norm) || std::abs(norm - 1) > .001 ||
        !std::isfinite(transform.translation.x) || !std::isfinite(transform.translation.y) || !std::isfinite(transform.translation.z))
        return std::unexpected(operation_error{"invalid sensor destination geometry"});
    const auto n = (grid.side + 7) / 8;
    if (observed.size() != std::size_t(n) * n * n)
        return std::unexpected(operation_error{"invalid sensor observation mask"});
    std::set<int3, decltype(&block_index_less)> result(&block_index_less);
    const double tile_size = double(size) * 8;

    for (std::size_t i = 0; i < observed.size(); ++i)
    {
        if (!observed[i]) continue;
        const unsigned cell[]{unsigned(i % n) * 8, unsigned(i / n % n) * 8, unsigned(i / (n * n)) * 8};
        float lower[]{INFINITY, INFINITY, INFINITY}, upper[]{-INFINITY, -INFINITY, -INFINITY};

        for (unsigned corner = 0; corner < 8; ++corner) {
            float p[3]{};

            for (unsigned axis = 0; axis < 3; ++axis)
                p[axis] = grid.origin + (float(cell[axis]) + (corner & (1U << axis) ? 9.f : -1.f)) * size;
            const auto point = transform_point(transform, {p[0], p[1], p[2]});
            const float value[]{point.x, point.y, point.z};

            for (unsigned axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(value[axis])) return std::unexpected(operation_error{"destination transform exceeds range"});
                lower[axis] = std::min(lower[axis], value[axis]);
                 upper[axis] = std::max(upper[axis], value[axis]);
            }
        }
        int lo[3]{}, hi[3]{};
        std::size_t volume{1};

        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const double a = std::floor(lower[axis] / tile_size), b = std::floor(upper[axis] / tile_size);

            if (!std::isfinite(a) || !std::isfinite(b) || a < INT_MIN / 8 + 1 || b > INT_MAX / 8 - 1)
                return std::unexpected(operation_error{"destination tile index exceeds range"});
            lo[axis] = static_cast<int>(a); hi[axis] = static_cast<int>(b);
            const auto extent = std::size_t(hi[axis] - lo[axis]) + 1;
            if (extent > capacity / volume) return std::unexpected(operation_error{"destination tiles exceed capacity"});
            volume *= extent;
        }

        for (int z = lo[2]; z <= hi[2]; ++z)
            for (int y = lo[1]; y <= hi[1]; ++y)
                for (int x = lo[0]; x <= hi[0]; ++x) {
                    result.insert({x, y, z});
                    if (result.size() > capacity) return std::unexpected(operation_error{"destination tiles exceed capacity"});
                }
    }
    return std::vector<int3>(result.begin(), result.end());
}
}
