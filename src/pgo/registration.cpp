#include <covoxslam/pgo.hpp>
#include <covoxslam/voxel/flags.hpp>
#include <cmath>

namespace covoxslam::pgo
{
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
    problem.registration.evaluator = ceronoceres::RegistrationModel{}.evaluator;
    problem.registration.loss = options.loss;
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
            ceronoceres::TsdfTile tile;
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
            ceronoceres::EsdfTile tile;
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
        ceronoceres::RegistrationGroup group;
        group.node_i = direction.source; group.node_j = direction.target;
        group.volume_i = direction.source; group.volume_j = direction.target;
        group.field = ceronoceres::FieldKind::Esdf;
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
