#pragma once

#include <covoxslam/sensor_grid_sampling.h>
#include "../sparse_mapping.h"

namespace covoxslam::sensor_grid
{
int cuda_sensor_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    std::uint32_t* masks, void* stream, unsigned* launches = nullptr);
int cuda_sensor_mask(field_view source, std::uint8_t* flags, void* stream);
int cuda_sensor_compact_mask(field_view source, const std::uint32_t* offsets,
    const std::uint32_t* cells, std::uint8_t* flags, void* stream);
int cuda_sensor_masked_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    const std::uint32_t* masks, unsigned long long* updates, unsigned long long* candidates,
    void* stream, bool compacted_control);
int cuda_sensor_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose local_from_sensor, float size, float maximum_weight,
    float surface_band, unsigned long long* updates, void* stream, bool fp32 = false, bool specialized = false,
    float truncation = FLT_MAX);
}
