#include <covoxslam/pgo.hpp>
#include <algorithm>
#include <cmath>

namespace covoxslam::pgo
{
namespace
{
void include_point(surface_bounds& bounds, std::array<double, 3> point)
{
    if (!bounds.valid) { bounds.minimum = point; bounds.maximum = point; bounds.valid = true; }

    for (std::size_t axis = 0; axis < 3; ++axis) {
        bounds.minimum[axis] = std::min(bounds.minimum[axis], point[axis]);
        bounds.maximum[axis] = std::max(bounds.maximum[axis], point[axis]);
    }
}
bool overlaps(const surface_bounds& left, const surface_bounds& right, double padding)
{
    if (!left.valid || !right.valid) return false;

    for (std::size_t axis = 0; axis < 3; ++axis)
        if (left.minimum[axis] - right.maximum[axis] > padding ||
            right.minimum[axis] - left.maximum[axis] > padding) return false;
    return true;
}
}
operation_result<candidate_set> find_candidates(std::span<const submap> maps, candidate_options options)
{
    if (!options.sample_stride || !std::isfinite(options.overlap_padding_m) || options.overlap_padding_m < 0)
        return std::unexpected(operation_error{"invalid registration candidate settings"});
    candidate_set result;
    result.bounds.resize(maps.size());
    std::vector<std::vector<surface_sample>> samples(maps.size());

    for (std::size_t index = 0; index < maps.size(); ++index)
    {
        const auto& map = maps[index];
        const auto& pose = map.frame.anchor;

        if (!map.config.valid() || !std::isfinite(pose.x) || !std::isfinite(pose.y) ||
            !std::isfinite(pose.z) || !std::isfinite(pose.yaw))
            return std::unexpected(operation_error{"invalid registration source submap"});
        std::vector<int3> blocks;
        blocks.reserve(map.tsdf.size());

        for (const auto& entry : map.tsdf) blocks.push_back(entry.first);
        std::sort(blocks.begin(), blocks.end(), block_index_less);
        const double cosine = std::cos(pose.yaw), sine = std::sin(pose.yaw), size = map.config.voxel_size;
        std::uint64_t ordinal{}, eligible{};

        for (const auto block : blocks) {
            const auto& tile = map.tsdf.find(block)->second;

            for (std::size_t local = 0; local < tile_voxels; ++local, ++ordinal)
            {
                const auto& value = tile.voxels[local];

                if (!(value.flags & observed_flag) || !(value.flags & near_surface_flag) ||
                    !(value.weight > 0) || value.weight < map.config.esdf_minimum_weight) continue;
                if (!std::isfinite(value.distance) || !std::isfinite(value.weight))
                    return std::unexpected(operation_error{"nonfinite registration source voxel"});
                const ceronoceres::Vec3d p{
                    size * (double(block.x) * tile_edge + local % tile_edge + 0.5),
                    size * (double(block.y) * tile_edge + (local / tile_edge) % tile_edge + 0.5),
                    size * (double(block.z) * tile_edge + local / (tile_edge * tile_edge) + 0.5)};
                const std::array<double, 3> world{pose.x + cosine * p.x - sine * p.y,
                    pose.y + sine * p.x + cosine * p.y, pose.z + p.z};

                if (!std::all_of(world.begin(), world.end(), [](double x) { return std::isfinite(x); }))
                    return std::unexpected(operation_error{"registration bounds exceed finite coordinates"});
                include_point(result.bounds[index], world);
                if (eligible++ % options.sample_stride == 0)
                    samples[index].push_back({p, value.distance, value.weight, ordinal});
            }
        }
    }
    std::uint64_t diagnostic = 1;

    for (std::size_t left = 0; left < maps.size(); ++left) {

        for (std::size_t right = left + 1; right < maps.size(); ++right)
        {
            if (!overlaps(result.bounds[left], result.bounds[right], options.overlap_padding_m)) continue;
            if (!samples[left].empty()) result.directions.push_back({left, right, diagnostic++, samples[left]});
            if (!samples[right].empty()) result.directions.push_back({right, left, diagnostic++, samples[right]});
        }
    }
    return result;
}
}
