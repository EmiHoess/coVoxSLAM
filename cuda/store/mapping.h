#pragma once
#include <covoxslam/storage/device_view.hpp>
#include <covoxslam/store/device_index.hpp>
#include <covoxslam/sensor_grid_sampling.h>
#include "../sparse_mapping.h"
#include <covoxslam/store/band_esdf.hpp>

namespace covoxslam::store
{
int run_band_esdf(storage::device_descriptor source, device_index_view source_index,
    storage::device_descriptor target, device_index_view target_index,
    storage::device_descriptor work, const block_key* keys, unsigned count,
    detail::cuda_tile_coordinate* coordinates, void* slots, unsigned* status,
    band_esdf_options options, band_esdf_metrics* metrics, void* stream);
int pack_esdf_input(storage::device_descriptor payload, device_index_view index,
    const block_key* keys, std::uint32_t count, detail::cuda_tile_coordinate* ordered_keys,
    detail::cuda_tsdf_tile* values, unsigned* status, void* stream);
int unpack_esdf_output(storage::device_descriptor payload, device_index_view index,
    const block_key* keys, std::uint32_t count, const void* values,
    storage::schema schema, unsigned* status, void* stream);
int integrate_ordered(storage::device_descriptor payload, device_index_view index,
    const detail::cuda_ray_contribution* samples, std::size_t sample_count,
    const detail::cuda_voxel_segment* segments, std::uint32_t segment_count,
    detail::cuda_fusion_parameters config, unsigned* status, void* stream);
int integrate_nearest(storage::device_descriptor payload, device_index_view index,
    sensor_grid::field_view source, const block_key* keys, std::uint32_t count,
    detail::cuda_projective_pose transform, float size, float weight, float band,
    unsigned* status, void* stream);
}
