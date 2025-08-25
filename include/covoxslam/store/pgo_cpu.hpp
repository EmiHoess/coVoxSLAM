#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/store/pgo_fields.hpp>
#if defined(covoxslam_has_pgo)
namespace covoxslam::store
{
inline operation_result<ceronoceres::PreparedProblem> prepare_cpu_fields(pgo::graph_snapshot graph,
    std::shared_ptr<const ceronoceres::field_snapshot> fields, pgo_field_mode mode)
{
    if (!fields || fields->device != -1 || mode == pgo_field_mode::resident_cuda)
        return std::unexpected(operation_error{"CPU field mode is unsupported"});
    if (mode == pgo_field_mode::borrowed_cpu) return prepare_borrowed_pgo(std::move(graph), std::move(fields));
    if (mode != pgo_field_mode::copied) return std::unexpected(operation_error{"unknown PGO field mode"});
    const auto error = ceronoceres::validate_fields(graph.problem, *fields);
    if (!error.empty()) return std::unexpected(operation_error{error});

    for (const auto& tile : fields->tiles) {
        const auto copy = [&]<class tile_type>(std::vector<tile_type>& output) {
            tile_type result;
            result.volume = tile.volume; result.x = tile.x; result.y = tile.y; result.z = tile.z;

            for (std::size_t i = 0; i < 512; ++i) {
                double distance{}, weight{};
                const bool valid = ceronoceres::read_field(tile, i, distance, weight);
                result.voxels[i].distance = float(distance);
                result.voxels[i].flags = valid ? 1U : 0U;
                if constexpr (requires { result.voxels[i].weight; }) result.voxels[i].weight = float(weight);
            }
            output.push_back(std::move(result));
        };
        switch (tile.kind) {
        case ceronoceres::FieldKind::Tsdf: copy(graph.problem.tsdf_tiles); break;
        case ceronoceres::FieldKind::Esdf: copy(graph.problem.esdf_tiles); break;
        }
    }
    return pgo::prepare_graph(std::move(graph));
}
template<storage::layout l, storage::schema es>
operation_result<ceronoceres::PreparedProblem> prepare_cpu_snapshot(pgo::graph_snapshot graph,
    const geometry_snapshot<l, es>& snapshot, std::span<const pgo_field_binding> bindings, pgo_field_mode mode)
{
    auto checked = validate_pgo_bindings(graph, snapshot, bindings);
    if (!checked) return std::unexpected(checked.error());
    auto fields = borrow_pgo_fields(snapshot, bindings);
    if (!fields) return std::unexpected(fields.error());
    return prepare_cpu_fields(std::move(graph), std::move(*fields), mode);
}
}
#endif
