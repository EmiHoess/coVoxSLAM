#pragma once
#include <covoxslam/store/geometry.hpp>
#include <covoxslam/storage/transfer_plan.hpp>
#include <covoxslam/pgo.hpp>
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

    for (const auto& binding : bindings) {
        const auto* map = snapshot.find(binding.submap);
        if (!map) return std::unexpected(operation_error{"PGO submap is not captured"});
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
}
#endif
