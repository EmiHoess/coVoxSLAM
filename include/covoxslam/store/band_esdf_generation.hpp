#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/authority.hpp>

namespace covoxslam::store
{
inline operation_status validate_band_generation(const authority_snapshot& state)
{
    auto checked = require_current_esdf(state, 2);

    if (!checked) return checked;
    if (state.state == lifecycle::retired || state.writer != writer_kind::none)
        return std::unexpected(operation_error{"ESDF generation is not available"});
    return {};
}
inline operation_status validate_band_generation(const authority_snapshot& state, const epochs& expected)
{
    auto checked = validate_band_generation(state);
    if (!checked) return checked;

    if (!expected.tsdf || expected.esdf_source != expected.tsdf ||
        state.content[0] != expected.tsdf || state.content[1] != expected.esdf ||
        state.esdf_source != expected.esdf_source)
        return std::unexpected(operation_error{"ESDF staging source generation is stale"});
    return {};
}
}
