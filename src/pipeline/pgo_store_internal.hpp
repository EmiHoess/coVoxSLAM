#pragma once
#include <covoxslam/core/byte_units.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/map_store/detail/geometry.hpp>
#include <covoxslam/storage/transfers.hpp>
#include <covoxslam/pgo/pose_graph.hpp>
#include <covoxslam/pgo/solver.hpp>
#include "../pgo/contracts.hpp"
#include <bit>
#include <chrono>
#include <mutex>
#include <ostream>
#if defined(covoxslam_has_pgo)
#include <ceronoceres/field_views.hpp>

namespace covoxslam::store
{
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
        if (node.id != binding.submap || !pgo::same_pose(node.initial, *pose) ||
            volume.voxel_size != double(snapshot.parameters().voxel_size) ||
            (volume.has_tsdf && !map->tsdf) || (volume.has_esdf && !map->esdf))
            return std::unexpected(operation_error{"PGO graph and field snapshot differ"});
    }
    return {};
}
#if defined(covoxslam_has_cuda)
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
#endif
struct preparation_key
{
    std::vector<std::byte> bytes;
    bool operator==(const preparation_key&) const = default;
};
inline operation_result<preparation_key> make_preparation_key(const pgo::graph_snapshot& graph,
    const ceronoceres::field_snapshot& fields, std::size_t budget)
{
    if (!graph.problem.tsdf_tiles.empty() || !graph.problem.esdf_tiles.empty())
        return std::unexpected(operation_error{"field cache requires uncopied graph payloads"});
    preparation_key key;
    bool overflow{};
    const auto add = [&]<class t>(t value) {
        if (overflow || sizeof(t) > budget - key.bytes.size()) { overflow = true; return; }
        const auto bytes = std::bit_cast<std::array<std::byte, sizeof(t)>>(value);
        key.bytes.insert(key.bytes.end(), bytes.begin(), bytes.end());
    };
    const auto pose = [&](const auto& p) { add(p.x); add(p.y); add(p.z); add(p.yaw); };
    const auto loss = [&](const auto& l) { add(l.kind); add(l.scale); };
    const auto& p = graph.problem;
    add(ceronoceres::field_view_api_version);
    add(fields.version); add(fields.device);
    add(fields.dependencies.size());
    for (auto dependency : fields.dependencies) add(dependency);
    add(graph.node_ids.size());

    for (auto id : graph.node_ids) add(id);
    add(p.nodes.size());
    for (const auto& node : p.nodes) { add(node.id); pose(node.initial); add(node.constant); }
    add(p.relative_factors.size());

    for (const auto& factor : p.relative_factors) {
        add(factor.node_i); add(factor.node_j); pose(factor.measured);
        for (auto value : factor.sqrt_information) add(value);
        loss(factor.loss); add(factor.diagnostic_id);
    }
    add(p.absolute_factors.size());

    for (const auto& factor : p.absolute_factors) {
        add(factor.reference_node); add(factor.measured_node); pose(factor.measured_relative);
        for (auto value : factor.sqrt_information) add(value);
        loss(factor.loss); add(factor.diagnostic_id);
    }
    add(p.volumes.size());
    for (const auto& volume : p.volumes) {
        add(volume.pose_node); add(volume.voxel_size); add(volume.has_tsdf); add(volume.has_esdf);
    }
    add(p.registration_groups.size());

    for (const auto& group : p.registration_groups) {
        add(group.node_i); add(group.node_j); add(group.volume_i); add(group.volume_j); add(group.field);
        add(group.sample_begin); add(group.sample_count); add(group.direction_weight); add(group.diagnostic_id);
    }
    add(p.registration_samples.size());
    for (const auto& sample : p.registration_samples) {
        add(sample.group); add(sample.reference_local_point.x); add(sample.reference_local_point.y);
        add(sample.reference_local_point.z); add(sample.reference_distance); add(sample.reference_tsdf_weight); add(sample.ordinal);
    }
    const auto& model = p.registration;
    add(model.evaluator); loss(model.loss); add(model.minimum_tsdf_weight); add(model.maximum_confidence_weight);
    add(model.minimum_gradient_norm); add(model.maximum_absolute_residual_m); add(model.allow_nearest_fallback);
    add(fields.tiles.size());

    for (const auto& tile : fields.tiles) {
        add(tile.volume); add(tile.kind); add(tile.x); add(tile.y); add(tile.z);
        add(std::uintptr_t(tile.distance)); add(std::uintptr_t(tile.weight)); add(std::uintptr_t(tile.flags));
        add(tile.voxel_stride); add(tile.encoding); add(tile.voxel_size); add(tile.maximum_distance);
        add(tile.content_epoch); add(tile.allocation_generation); add(tile.slot_generation);
    }
    if (overflow) return std::unexpected(operation_error{"preparation key exceeds byte budget"});
    return key;
}
struct preparation_metrics
{
    std::size_t hits{}, misses{}, invalidations{}, key_bytes{};
    double key_ms{}, preparation_ms{};
};
struct cached_preparation
{
    ceronoceres::PreparedProblem problem;
    bool reused{};
};
inline operation_status write_preparation_metrics(std::ostream& output, const preparation_metrics& metrics,
    const ceronoceres::SolverTelemetry& solver)
{
    output << "preparation cache hits " << metrics.hits << ", misses " << metrics.misses
        << ", invalidations " << metrics.invalidations << ", key bytes " << metrics.key_bytes
        << ", key ms " << metrics.key_ms << ", prepare ms " << metrics.preparation_ms << '\n'
        << "solver upload bytes " << solver.host_to_device_bytes << ", download bytes " << solver.device_to_host_bytes
        << ", upload ms " << solver.upload_ms << ", total ms " << solver.total_ms << '\n';
    if (!output) return std::unexpected(operation_error{"preparation report write failed"});
    return {};
}
inline operation_result<ceronoceres::SolveResult> solve_prepared_fields(const cached_preparation& prepared,
    const pgo::options& execution, const ceronoceres::NativeLmOptions& lm = {})
{
    switch (execution.solver) {
    case pgo::backend::native_cpu: return pgo::solve_cpu(prepared.problem, execution, lm);
    case pgo::backend::cuda_cold: {
        auto configured = pgo::cuda_solve_options(execution, lm);
        if (!configured) return std::unexpected(configured.error());
        auto solved = ceronoceres::solve_cuda_checked(prepared.problem, configured->common,
            std::get<ceronoceres::NativeCudaColdOptions>(configured->backend_options));
        if (const auto* error = std::get_if<ceronoceres::cuda_solve_error>(&solved))
            return std::unexpected(operation_error{error->message});
        auto result = std::get<ceronoceres::SolveResult>(std::move(solved));
        auto valid = pgo::validate_solve_result(result, prepared.problem.node_count(), ceronoceres::Backend::NativeCudaCold);
        if (!valid) return std::unexpected(valid.error());
        return result;
    }
    default: return std::unexpected(operation_error{"unsupported field solver"});
    }
}
class preparation_cache
{
    static constexpr std::size_t default_maximum_key_bytes = 64U * bytes_per_mebibyte;
    mutable std::mutex mutex_;
    std::optional<preparation_key> key_;
    ceronoceres::PreparedProblem problem_;
    preparation_metrics metrics_;
    std::size_t maximum_key_bytes_;
public:
    explicit preparation_cache(std::size_t maximum_key_bytes = default_maximum_key_bytes)
        : maximum_key_bytes_(maximum_key_bytes) {}
    preparation_metrics metrics() const
    {
        std::lock_guard lock(mutex_);
        return metrics_;
    }
    void invalidate()
    {
        std::lock_guard lock(mutex_);
        key_.reset();
        problem_ = {};
        metrics_.key_bytes = 0;
        ++metrics_.invalidations;
    }
    operation_result<cached_preparation> prepare(pgo::graph_snapshot graph,
        std::shared_ptr<const ceronoceres::field_snapshot> fields)
    {
        const auto started = std::chrono::steady_clock::now();
        auto checked = pgo::validate_graph(graph);
        if (!checked) return std::unexpected(checked.error());
        if (!fields) return std::unexpected(operation_error{"cache field snapshot is absent"});
        const auto error = ceronoceres::validate_fields(graph.problem, *fields);
        if (!error.empty()) return std::unexpected(operation_error{error});
        auto key = make_preparation_key(graph, *fields, maximum_key_bytes_);
        if (!key) return std::unexpected(key.error());
        const auto generated = std::chrono::steady_clock::now();
        std::lock_guard lock(mutex_);
        metrics_.key_ms += std::chrono::duration<double, std::milli>(generated - started).count();

        if (key_ && *key_ == *key) {
            ++metrics_.hits;
            return cached_preparation{problem_, true};
        }
        auto prepared = ceronoceres::prepare_fields(std::move(graph.problem), std::move(fields));
        if (const auto* failure = std::get_if<std::string>(&prepared))
            return std::unexpected(operation_error{*failure});
        problem_ = std::get<ceronoceres::PreparedProblem>(std::move(prepared));
        metrics_.preparation_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - generated).count();
        ++metrics_.misses;
        metrics_.key_bytes = key->bytes.size();
        key_ = std::move(*key);
        return cached_preparation{problem_, false};
    }
};
}
#endif
