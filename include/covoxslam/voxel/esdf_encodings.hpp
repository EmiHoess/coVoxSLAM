#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace covoxslam::storage
{
struct blocks_esdf_value
{
    float distance_m{};
    std::int32_t parent_dx{}, parent_dy{}, parent_dz{};
    std::uint32_t flags{};
};

struct projective_esdf_value
{
    float squared_distance_voxels{};
    std::int32_t parent_x{}, parent_y{}, parent_z{};
    std::uint32_t flags{};
};

static_assert(std::is_aggregate_v<blocks_esdf_value>);
static_assert(std::is_standard_layout_v<blocks_esdf_value>);
static_assert(std::is_trivially_copyable_v<blocks_esdf_value>);
static_assert(sizeof(blocks_esdf_value) == 20);
static_assert(alignof(blocks_esdf_value) == alignof(float));
static_assert(offsetof(blocks_esdf_value, distance_m) == 0);
static_assert(offsetof(blocks_esdf_value, parent_dx) == 4);
static_assert(offsetof(blocks_esdf_value, parent_dy) == 8);
static_assert(offsetof(blocks_esdf_value, parent_dz) == 12);
static_assert(offsetof(blocks_esdf_value, flags) == 16);

static_assert(std::is_aggregate_v<projective_esdf_value>);
static_assert(std::is_standard_layout_v<projective_esdf_value>);
static_assert(std::is_trivially_copyable_v<projective_esdf_value>);
static_assert(sizeof(projective_esdf_value) == 20);
static_assert(alignof(projective_esdf_value) == alignof(float));
static_assert(offsetof(projective_esdf_value, squared_distance_voxels) == 0);
static_assert(offsetof(projective_esdf_value, parent_x) == 4);
static_assert(offsetof(projective_esdf_value, parent_y) == 8);
static_assert(offsetof(projective_esdf_value, parent_z) == 12);
static_assert(offsetof(projective_esdf_value, flags) == 16);
}

namespace covoxslam::detail
{
struct cuda_site_esdf_voxel 
{
    float squared_distance_voxels{};
    std::int32_t parent[3]{};
   std::uint32_t flags{};
};

inline constexpr std::uint32_t cuda_esdf_observed = 1;
inline constexpr std::uint32_t cuda_esdf_inside = 2;
inline constexpr std::uint32_t cuda_esdf_site = 4;
inline constexpr std::uint32_t cuda_esdf_parent_assigned = 8;

static_assert(std::is_aggregate_v<cuda_site_esdf_voxel>);
static_assert(std::is_standard_layout_v<cuda_site_esdf_voxel>);
static_assert(std::is_trivially_copyable_v<cuda_site_esdf_voxel>);
static_assert(sizeof(cuda_site_esdf_voxel) == 20);
static_assert(alignof(cuda_site_esdf_voxel) == alignof(float));
static_assert(offsetof(cuda_site_esdf_voxel, squared_distance_voxels) == 0);
static_assert(offsetof(cuda_site_esdf_voxel, parent) == 4);
static_assert(offsetof(cuda_site_esdf_voxel, flags) == 16);

static_assert(cuda_esdf_observed == 1);
static_assert(cuda_esdf_inside == 2);
static_assert(cuda_esdf_site == 4);
static_assert(cuda_esdf_parent_assigned == 8);
static_assert((cuda_esdf_observed & cuda_esdf_inside) == 0);
static_assert((cuda_esdf_observed & cuda_esdf_site) == 0);
static_assert((cuda_esdf_observed & cuda_esdf_parent_assigned) == 0);
static_assert((cuda_esdf_inside & cuda_esdf_site) == 0);
static_assert((cuda_esdf_inside & cuda_esdf_parent_assigned) == 0);
static_assert((cuda_esdf_site & cuda_esdf_parent_assigned) == 0);

}
