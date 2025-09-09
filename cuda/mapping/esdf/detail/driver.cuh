#pragma once
#include <covoxslam/storage/device_view.hpp>
#include <covoxslam/map_store/device_index.hpp>
#include <covoxslam/mapping/tsdf/detail/sampling.hpp>
#include "../../detail/types.cuh"
#include <covoxslam/mapping/esdf/band_esdf.hpp>

namespace covoxslam::store
{
int run_band_esdf(storage::device_descriptor source, device_index_view source_index,
    storage::device_descriptor target, device_index_view target_index,
    storage::device_descriptor work, const block_key* keys, unsigned count,
    detail::cuda_tile_coordinate* coordinates, void* slots, unsigned* status,
    band_esdf_options options, band_esdf_metrics* metrics, void* stream);
}
