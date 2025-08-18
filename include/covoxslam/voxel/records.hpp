#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace covoxslam
{
struct tsdf_record
{
    float distance{};
    float weight{};
     std::uint32_t sample_count{};
    std::uint32_t flags{};
};

struct esdf_record {
    float signed_distance_m{};
    std::int32_t parent_dx{};
    std::int32_t parent_dy{};
    std::int32_t parent_dz{};
     std::uint32_t flags{};
}
;

using tsdf_voxel = tsdf_record;
using esdf_voxel = esdf_record;

static_assert(sizeof(float) == 4);
static_assert(std::is_aggregate_v<tsdf_record>);
static_assert(std::is_standard_layout_v<tsdf_record>);
static_assert(std::is_trivially_copyable_v<tsdf_record>);
static_assert(sizeof(tsdf_record) == 16);
static_assert(alignof(tsdf_record) == alignof(float));
static_assert(offsetof(tsdf_record, distance) == 0);
static_assert(offsetof(tsdf_record, weight) == 4);
static_assert(offsetof(tsdf_record, sample_count) == 8);
static_assert(offsetof(tsdf_record, flags) == 12);

static_assert(std::is_aggregate_v<esdf_record>);
static_assert(std::is_standard_layout_v<esdf_record>);
static_assert(std::is_trivially_copyable_v<esdf_record>);
static_assert(sizeof(esdf_record) == 20);
static_assert(alignof(esdf_record) == alignof(float));
static_assert(offsetof(esdf_record, signed_distance_m) == 0);
static_assert(offsetof(esdf_record, parent_dx) == 4);
static_assert(offsetof(esdf_record, parent_dy) == 8);
static_assert(offsetof(esdf_record, parent_dz) == 12);
static_assert(offsetof(esdf_record, flags) == 16);
}
