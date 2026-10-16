#include <covoxslam/pgo/registration.hpp>
#include <covoxslam/voxel/flags.hpp>
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
                const abseil::vector3 p{
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
operation_status append_registration(graph_snapshot& graph, std::span<const submap> maps,
    const candidate_set& candidates, registration_options options, bool copy_fields)
{
    const auto valid = validate_graph(graph);
    if (!valid) return valid;
    const auto valid_candidates = validate_candidates(candidates, maps.size());

    if (!valid_candidates) return valid_candidates;
    if (options.nearest_fallback)
        return std::unexpected(operation_error{"pose graph registration requires interpolated field support"});
    if (maps.size() != graph.node_ids.size() || !graph.problem.volumes.empty() ||
        !graph.problem.registration_groups.empty())
        return std::unexpected(operation_error{"registration fields already assigned or submap count differs"});
    auto staging = graph;
    auto& problem = staging.problem;
    problem.registration.profile = abseil::registration_profile::uniform_trilinear;
    problem.registration.robust_loss = options.loss;
    problem.registration.minimum_tsdf_weight = options.minimum_weight;
    problem.registration.maximum_confidence_weight = options.maximum_weight;
    problem.registration.minimum_gradient_norm = options.minimum_gradient;
    problem.registration.maximum_absolute_residual_m = options.maximum_residual_m;
    problem.registration.allow_nearest_fallback = options.nearest_fallback;

    for (std::size_t index = 0; index < maps.size(); ++index)
    {
        const auto& map = maps[index];

        if (map.frame.id != graph.node_ids[index] || !map.config.valid())
            return std::unexpected(operation_error{"registration submap identity or configuration differs"});
        problem.volumes.push_back({index, map.config.voxel_size, !map.tsdf.empty(), !map.esdf.empty()});
        if (!copy_fields) continue;

        for (const auto& entry : map.tsdf) {
            abseil::tsdf_tile tile;
            tile.volume = index;
            tile.x = entry.first.x; tile.y = entry.first.y; tile.z = entry.first.z;

            for (std::size_t local = 0; local < tile_voxels; ++local) {
                const auto& value = entry.second.voxels[local];

                if ((value.flags & observed_flag) && (!std::isfinite(value.distance) ||
                    !std::isfinite(value.weight) || value.weight <= 0))
                    return std::unexpected(operation_error{"invalid observed TSDF field value"});
                tile.voxels[local] = {value.distance, value.weight, value.flags};
            }
            problem.tsdf_tiles.push_back(std::move(tile));
        }

        for (const auto& entry : map.esdf)
        {
            abseil::esdf_tile tile;
            tile.volume = index;
            tile.x = entry.first.x; tile.y = entry.first.y; tile.z = entry.first.z;

            for (std::size_t local = 0; local < tile_voxels; ++local) {
                const auto& value = entry.second.voxels[local];
                if ((value.flags & observed_flag) && !std::isfinite(value.signed_distance_m))
                    return std::unexpected(operation_error{"invalid observed ESDF field value"});
                tile.voxels[local] = {value.signed_distance_m, value.flags};
            }
            problem.esdf_tiles.push_back(std::move(tile));
        }
    }

    for (const auto& direction : candidates.directions) {
        if (!problem.volumes[direction.source].has_tsdf || !problem.volumes[direction.target].has_esdf)
            return std::unexpected(operation_error{"registration direction has no supporting field"});
        abseil::registration_group group;
        group.node_i = direction.source; group.node_j = direction.target;
        group.volume_i = direction.source; group.volume_j = direction.target;
        group.field = abseil::field_kind::esdf;
        group.sample_begin = problem.registration_samples.size();
        group.sample_count = direction.samples.size();
        group.diagnostic_id = direction.diagnostic_id;

        for (const auto& sample : direction.samples)
            problem.registration_samples.push_back({problem.registration_groups.size(), sample.point,
                sample.distance, sample.weight, sample.ordinal});
        problem.registration_groups.push_back(group);
    }
    const auto complete = validate_graph(staging);
    if (!complete) return complete;
    graph = std::move(staging);
    return {};
}
}
