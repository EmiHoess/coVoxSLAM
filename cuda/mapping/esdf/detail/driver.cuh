#pragma once
#include <covoxslam/storage/views.hpp>
#include <covoxslam/map_store/device_index.hpp>
#include <covoxslam/mapping/tsdf/detail/sampling.hpp>
#include "../../detail/types.cuh"
#include <covoxslam/mapping/esdf/band_esdf.hpp>

namespace covoxslam::store
{
struct band_tile_slot
{
    unsigned source{}, target{};
};
struct band_status
{
    unsigned invalid{}, changed{};
};
inline constexpr unsigned band_invalid_geometry_mask = 1U;
inline constexpr unsigned band_invalid_value_mask = 2U;
inline constexpr unsigned band_changed_mask = 1U;
static_assert(band_invalid_geometry_mask == 1U);
static_assert(band_invalid_value_mask == 2U);
static_assert(band_changed_mask == 1U);
static_assert((band_invalid_geometry_mask & band_invalid_value_mask) == 0);
int run_band_esdf(storage::device_descriptor source, device_index_view source_index,
    storage::device_descriptor target, device_index_view target_index,
    storage::device_descriptor work, const block_key* keys, unsigned count,
    detail::cuda_tile_coordinate* coordinates, band_tile_slot* slots, band_status* status,
    band_esdf_options options, band_esdf_metrics* metrics, void* stream);
}
