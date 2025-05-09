#pragma once
#include <covoxslam/store/pgo_fields.hpp>
#include <covoxslam/store/coordinator.hpp>
#if defined(covoxslam_has_pgo) && defined(covoxslam_has_cuda)
namespace covoxslam::store
{
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
}
#endif
