#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/map_store/detail/geometry.hpp>
#include <covoxslam/storage/transfers.hpp>
#include <covoxslam/optimization/pose_graph.hpp>
#include <covoxslam/optimization/solver.hpp>
#if defined(covoxslam_has_pgo)
#include <ceronoceres/field_views.hpp>

namespace covoxslam::store
{
enum class pgo_field_mode { copied, borrowed_cpu, resident_cuda };
struct pgo_field_binding
{
    std::uint64_t submap{};
    std::size_t volume{};
     epochs expected;
};
template<storage::schema s> ceronoceres::field_tile_view pgo_tile(
    storage::transfer_endpoint source, const slot_record& slot, std::size_t index,
    std::size_t volume, std::uint64_t content, std::uint64_t generation, geometry_parameters parameters)
{
    ceronoceres::field_tile_view tile;
    tile.volume = volume;
    tile.x = slot.key.x; tile.y = slot.key.y; tile.z = slot.key.z;
    tile.kind = s == storage::schema::tsdf ? ceronoceres::FieldKind::Tsdf : ceronoceres::FieldKind::Esdf;
    tile.distance = source.fields[0] + index * source.tile_stride;
    tile.flags = source.fields[storage::field_count<s> - 1] + index * source.tile_stride;
    if constexpr (s == storage::schema::tsdf) tile.weight = source.fields[1] + index * source.tile_stride;
    if constexpr (s == storage::schema::projective_esdf) tile.encoding = ceronoceres::field_encoding::squared_voxels;
    tile.voxel_stride = source.voxel_stride;
    tile.voxel_size = parameters.voxel_size;
    tile.maximum_distance = parameters.distance_limit;
    tile.content_epoch = content; tile.allocation_generation = generation;
    tile.slot_generation = slot.generation;
    return tile;
}
template<storage::layout l, storage::schema es>
operation_result<std::shared_ptr<const ceronoceres::field_snapshot>> borrow_pgo_fields(
    const geometry_snapshot<l, es>& snapshot, std::span<const pgo_field_binding> bindings)
{
    auto result = std::make_shared<ceronoceres::field_snapshot>();
    result->identity = snapshot.id();
    std::unordered_set<std::uint64_t> maps, volumes;

    for (const auto& binding : bindings) {
        const auto* map = snapshot.find(binding.submap);
        if (!map) return std::unexpected(operation_error{"PGO submap is not captured"});
        const auto& current = map->record.versions;
        const auto& expected = binding.expected;
        result->dependencies.insert(result->dependencies.end(), {binding.submap, binding.volume,
            current.pose, current.tsdf, current.esdf, current.esdf_source, current.allocation, current.descriptor,
            std::uint64_t(l), std::uint64_t(es)});
        if (!maps.insert(binding.submap).second || !volumes.insert(binding.volume).second ||
            current.tsdf != expected.tsdf || current.esdf != expected.esdf || current.pose != expected.pose ||
            current.allocation != expected.allocation || current.descriptor != expected.descriptor ||
            current.esdf_source != expected.esdf_source || (map->esdf && current.esdf_source != current.tsdf))
            return std::unexpected(operation_error{"PGO field binding is stale or duplicated"});
        const auto append = [&]<storage::schema s>(const std::shared_ptr<const field_pool<s, l>>& field) {
            if (!field) return;
            result->owners.push_back(field);
            const auto endpoint = storage::transfer_storage(field->payload);
            const auto slots = field->index.records();

            for (std::size_t i = 0; i < slots.size(); ++i)
                if (slots[i].occupied)
                    result->tiles.push_back(pgo_tile<s>(endpoint, slots[i], i, binding.volume,
                        field->content_epoch, field->payload.generation(), snapshot.parameters()));
        };
        append(map->tsdf);
        append(map->esdf);
    }
    return std::shared_ptr<const ceronoceres::field_snapshot>(std::move(result));
}
template<storage::layout l, storage::schema es>
operation_status validate_pgo_bindings(const pgo::graph_snapshot& graph,
    const geometry_snapshot<l, es>& snapshot, std::span<const pgo_field_binding> bindings)
{
    for (const auto& binding : bindings) {
        if (binding.volume >= graph.problem.volumes.size())
            return std::unexpected(operation_error{"PGO volume is absent"});
        const auto& volume = graph.problem.volumes[binding.volume];
        const auto* map = snapshot.find(binding.submap);
        if (!map || volume.pose_node >= graph.problem.nodes.size())
            return std::unexpected(operation_error{"PGO node is absent"});
        const auto& node = graph.problem.nodes[volume.pose_node];
        auto pose = snapshot.pose(binding.submap);
        if (!pose) return std::unexpected(pose.error());
        if (node.id != binding.submap || node.initial.x != pose->x || node.initial.y != pose->y ||
            node.initial.z != pose->z || std::abs(ceronoceres::wrap_yaw(node.initial.yaw - pose->yaw)) > 1e-12 ||
            volume.voxel_size != double(snapshot.parameters().voxel_size) ||
            (volume.has_tsdf && !map->tsdf) || (volume.has_esdf && !map->esdf))
            return std::unexpected(operation_error{"PGO graph and field snapshot differ"});
    }
    return {};
}
inline operation_result<ceronoceres::PreparedProblem> prepare_borrowed_pgo(pgo::graph_snapshot graph,
    std::shared_ptr<const ceronoceres::field_snapshot> fields)
{
    auto valid = pgo::validate_graph(graph);
    if (!valid) return std::unexpected(valid.error());
    if (!fields || fields->device != -1)
        return std::unexpected(operation_error{"CPU PGO requires a host field snapshot"});
    auto prepared = ceronoceres::prepare_fields(std::move(graph.problem), std::move(fields));
    if (const auto* error = std::get_if<std::string>(&prepared))
        return std::unexpected(operation_error{*error});
    return std::get<ceronoceres::PreparedProblem>(std::move(prepared));
}
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
#if defined(covoxslam_has_cuda)
class resident_pgo_problem
{
    ceronoceres::PreparedProblem prepared_;
    std::shared_ptr<const ceronoceres::field_snapshot> fields_;
    resident_pgo_problem(ceronoceres::PreparedProblem prepared,
        std::shared_ptr<const ceronoceres::field_snapshot> fields)
        : prepared_(std::move(prepared)), fields_(std::move(fields)) {}
public:
    static operation_result<resident_pgo_problem> create(pgo::graph_snapshot graph,
        std::shared_ptr<const ceronoceres::field_snapshot> fields)
    {
        auto valid = pgo::validate_graph(graph);
        if (!valid) return std::unexpected(valid.error());
        if (!fields || fields->device < 0)
            return std::unexpected(operation_error{"resident solve requires device fields"});
        auto retained = std::make_shared<const ceronoceres::field_snapshot>(*fields);
        auto prepared = ceronoceres::prepare_fields(std::move(graph.problem), retained);
        if (const auto* error = std::get_if<std::string>(&prepared))
            return std::unexpected(operation_error{*error});
        return resident_pgo_problem(std::get<ceronoceres::PreparedProblem>(std::move(prepared)), std::move(retained));
    }
    const ceronoceres::PreparedProblem& prepared() const noexcept { return prepared_; }
    operation_result<ceronoceres::SolveResult> solve(const pgo::options& execution,
        const ceronoceres::NativeLmOptions& lm = {}) const
    {
        auto configured = pgo::cuda_solve_options(execution, lm);
        if (!configured) return std::unexpected(configured.error());
        if (execution.device != fields_->device)
            return std::unexpected(operation_error{"resident solve device differs from field allocation"});
        const auto pin = fields_;
        auto result = ceronoceres::solve_cuda_checked(prepared_, configured->common,
            std::get<ceronoceres::NativeCudaColdOptions>(configured->backend_options));
        if (const auto* error = std::get_if<ceronoceres::cuda_solve_error>(&result))
            return std::unexpected(operation_error{error->message});
        auto solution = std::get<ceronoceres::SolveResult>(std::move(result));
        auto valid = pgo::validate_solve_result(solution, prepared_.node_count(), ceronoceres::Backend::NativeCudaCold);
        if (!valid) return std::unexpected(valid.error());
        return solution;
    }
};
template<storage::layout l, storage::schema es>
operation_result<std::shared_ptr<const ceronoceres::field_snapshot>> resident_pgo_fields(
    std::span<const resident_entry<l, es>> maps, std::span<const pgo_field_binding> bindings,
    geometry_parameters parameters)
{
    auto valid = validate_geometry(parameters);
    if (!valid) return std::unexpected(valid.error());
    auto identity = allocate_snapshot_id();
    if (!identity) return std::unexpected(identity.error());
    auto result = std::make_shared<ceronoceres::field_snapshot>();
    result->identity = *identity;
    std::unordered_set<std::size_t> volumes;
    std::unordered_set<std::uint64_t> ids;

    for (const auto& binding : bindings) {
        const auto found = std::find_if(maps.begin(), maps.end(), [&](const auto& map) { return map.record.id == binding.submap; });
        if (found == maps.end() || !ids.insert(binding.submap).second || !volumes.insert(binding.volume).second)
            return std::unexpected(operation_error{"resident PGO bindings are invalid"});
        const auto& current = found->record.versions;
        const auto& expected = binding.expected;
        result->dependencies.insert(result->dependencies.end(), {binding.submap, binding.volume,
            current.pose, current.tsdf, current.esdf, current.esdf_source, current.allocation, current.descriptor,
            std::uint64_t(l), std::uint64_t(es)});
        if (found->record.state != lifecycle::finalized || current.tsdf != expected.tsdf ||
            current.esdf != expected.esdf || current.esdf_source != expected.esdf_source || current.pose != expected.pose ||
            current.allocation != expected.allocation || current.descriptor != expected.descriptor)
            return std::unexpected(operation_error{"resident PGO binding is stale"});
        const auto append = [&]<storage::schema s>(const std::shared_ptr<const gpu_field<s, l>>& field) -> operation_status {
            if (!field) return {};
            auto descriptor = field->payload.descriptor();
            if (!descriptor) return std::unexpected(descriptor.error());
            auto topology = field->source_identity.lock();
            if (!topology || descriptor->size_tiles != topology->index.slot_count())
                return std::unexpected(operation_error{"resident PGO topology is unavailable"});
            if (result->device != -1 && result->device != descriptor->device)
                return std::unexpected(operation_error{"resident PGO spans multiple devices"});
            result->device = descriptor->device;
            result->owners.push_back(field);
            result->owners.push_back(topology);
            auto endpoint = storage::transfer_storage(*descriptor);
            const auto slots = topology->index.records();

            for (std::size_t i = 0; i < slots.size(); ++i)
                if (slots[i].occupied)
                    result->tiles.push_back(pgo_tile<s>(endpoint, slots[i], i, binding.volume,
                        field->content_epoch, field->allocation_generation, parameters));
            return {};
        };
        valid = append(found->tsdf);
        if (!valid) return std::unexpected(valid.error());
        valid = append(found->esdf);
        if (!valid) return std::unexpected(valid.error());
    }
    if (result->device < 0) return std::unexpected(operation_error{"resident PGO has no device fields"});
    return std::shared_ptr<const ceronoceres::field_snapshot>(std::move(result));
}
template<storage::layout l, storage::schema es>
operation_result<resident_pgo_problem> prepare_resident_snapshot(pgo::graph_snapshot graph,
    std::span<const resident_entry<l, es>> maps, std::span<const pgo_field_binding> bindings,
    geometry_parameters parameters)
{
    if (parameters.pose != geometry_pose::optimized)
        return std::unexpected(operation_error{"resident PGO requires optimized initial graph poses"});
    for (const auto& binding : bindings) {
        auto found = std::find_if(maps.begin(), maps.end(), [&](const auto& map) { return map.record.id == binding.submap; });
        if (found == maps.end() || binding.volume >= graph.problem.volumes.size())
            return std::unexpected(operation_error{"resident graph volume is absent"});
        const auto& volume = graph.problem.volumes[binding.volume];
        if (volume.pose_node >= graph.problem.nodes.size())
            return std::unexpected(operation_error{"resident graph node is absent"});
        const auto& node = graph.problem.nodes[volume.pose_node];
        const auto& pose = found->record.optimized_pose;
        if (node.id != binding.submap || node.initial.x != pose.x || node.initial.y != pose.y || node.initial.z != pose.z ||
            std::abs(ceronoceres::wrap_yaw(node.initial.yaw - pose.yaw)) > 1e-12 || volume.voxel_size != double(parameters.voxel_size) ||
            (volume.has_tsdf && !found->tsdf) || (volume.has_esdf && !found->esdf))
            return std::unexpected(operation_error{"resident graph and map snapshot differ"});
    }
    auto fields = resident_pgo_fields<l, es>(maps, bindings, parameters);
    if (!fields) return std::unexpected(fields.error());
    return resident_pgo_problem::create(std::move(graph), std::move(*fields));
}
#endif
}
#endif

#include <covoxslam/store/pgo_cache.hpp>
