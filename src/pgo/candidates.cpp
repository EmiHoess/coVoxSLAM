#include <covoxslam/pgo.hpp>
#include <algorithm>
#include <cmath>
#include <set>

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
bool has_field(const submap& map)
{
    for (const auto& entry : map.esdf)
        for (const auto& voxel : entry.second.voxels)
            if ((voxel.flags & observed_flag) && std::isfinite(voxel.signed_distance_m)) return true;
    return false;
}
}
operation_status validate_candidates(const candidate_set& candidates, std::size_t map_count)
{
    if (candidates.bounds.size() != map_count)
        return std::unexpected(operation_error{"candidate bounds differ from submap count"});
    std::set<std::pair<std::size_t, std::size_t>> pairs;
    std::set<std::uint64_t> identifiers;

    for (const auto& direction : candidates.directions) {

        if (direction.source >= map_count || direction.target >= map_count || direction.source == direction.target ||
            direction.samples.empty() || !pairs.emplace(direction.source, direction.target).second ||
            !identifiers.insert(direction.diagnostic_id).second)
            return std::unexpected(operation_error{"duplicate or unsupported registration direction"});
        std::optional<std::uint64_t> previous;

        for (const auto& sample : direction.samples) {

            if (!std::isfinite(sample.point.x) || !std::isfinite(sample.point.y) || !std::isfinite(sample.point.z) ||
                !std::isfinite(sample.distance) || !std::isfinite(sample.weight) || sample.weight <= 0 ||
                (previous && sample.ordinal <= *previous))
                return std::unexpected(operation_error{"invalid registration sample sequence"});
            previous = sample.ordinal;
        }
    }
    return {};
}
operation_result<candidate_statistics> summarize_candidates(const candidate_set& candidates)
{
    const auto valid = validate_candidates(candidates, candidates.bounds.size());
    if (!valid) return std::unexpected(valid.error());
    candidate_statistics result;
    std::set<std::pair<std::size_t, std::size_t>> pairs;

    for (const auto& bounds : candidates.bounds) result.surface_maps += bounds.valid;

    for (const auto& direction : candidates.directions) {
        pairs.emplace(std::min(direction.source, direction.target), std::max(direction.source, direction.target));
        result.samples += direction.samples.size();
    }
    result.overlap_pairs = pairs.size();
    result.directed_groups = candidates.directions.size();
     return result;
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
    std::vector<bool> supported;
    for (const auto& map : maps) supported.push_back(has_field(map));

    for (std::size_t left = 0; left < maps.size(); ++left) {

        for (std::size_t right = left + 1; right < maps.size(); ++right)
        {
            if (!overlaps(result.bounds[left], result.bounds[right], options.overlap_padding_m)) continue;
            if (!samples[left].empty() && supported[right]) result.directions.push_back({left, right, diagnostic++, samples[left]});
            if (!samples[right].empty() && supported[left]) result.directions.push_back({right, left, diagnostic++, samples[right]});
        }
    }
    const auto valid = validate_candidates(result, maps.size());
    if (!valid) return std::unexpected(valid.error());
    return result;
}
}
