#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/map_store/detail/geometry_fields.hpp>

namespace covoxslam::store
{
template<storage::layout l, storage::schema es> operation_result<geometry_snapshot<l, es>> capture_geometry(
    coordinated_store<l, es>& maps, std::span<const std::uint64_t> ids, field_mask fields,
    geometry_parameters parameters, std::size_t payload_budget, std::optional<std::size_t> readback_budget = {})
{
    auto checked = validate_geometry(parameters);
    if (!checked) return std::unexpected(checked.error());
    if (ids.empty()) return std::unexpected(operation_error{"geometry selection is empty"});
    
    operation_result<immutable_snapshot<l, es>> captured = std::unexpected(operation_error{"GPU readback is unavailable"});

    #if defined(covoxslam_has_cuda)
    if (readback_budget) captured = maps.lease_readback(ids, fields, payload_budget, *readback_budget);
    else
#endif

    if (!readback_budget) captured = maps.lease(ids, fields, payload_budget);
    if (!captured) return std::unexpected(captured.error());
    return geometry_snapshot<l, es>::create(std::move(*captured), parameters);
}
}
