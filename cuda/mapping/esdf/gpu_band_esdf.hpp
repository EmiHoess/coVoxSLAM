#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/mapping/esdf/band_esdf.hpp>

namespace covoxslam::store
{
operation_result<band_esdf_metrics> generate_band_esdf(
    coordinated_store<storage::layout::pool_soa>& maps, std::uint64_t id, band_esdf_options options);
}
