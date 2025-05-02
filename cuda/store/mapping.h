#pragma once
#include <covoxslam/storage/device_view.hpp>
#include <covoxslam/store/device_index.hpp>
#include <covoxslam/sensor_grid_sampling.h>
#include "../sparse_mapping.h"

namespace covoxslam::store
{
int integrate_ordered(storage::device_descriptor payload, device_index_view index,
    const detail::cuda_ray_contribution* samples, std::size_t sample_count,
    const detail::cuda_voxel_segment* segments, std::uint32_t segment_count,
    detail::cuda_fusion_parameters config, unsigned* status, void* stream);
int integrate_nearest(storage::device_descriptor payload, device_index_view index,
    sensor_grid::field_view source, const block_key* keys, std::uint32_t count,
    detail::cuda_projective_pose transform, float size, float weight, float band,
    unsigned* status, void* stream);
}
