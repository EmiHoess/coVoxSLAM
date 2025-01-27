#pragma once
#include "map_native.h"

namespace covoxslam::sensor_grid
{
struct candidate_list_counters
{
    unsigned long long count{};
    unsigned overflow{}, invalid{};
};
static_assert(sizeof(candidate_list_counters) == 16);
struct candidate_list_view
{
    std::uint32_t* masks{};
    std::uint32_t* ids{};
    candidate_list_counters* counters{};
    std::uint32_t capacity{};
};
int cuda_sensor_append_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    candidate_list_view list, void* stream);
int cuda_sensor_list_candidates(field_view source, interpolation mode, const std::uint32_t* active,
    std::uint32_t active_count, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, detail::cuda_projective_pose transform,
    candidate_list_view list, void* stream);
int cuda_sensor_list_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    candidate_list_view list, unsigned long long* updates, unsigned long long* candidates, void* stream);
}
