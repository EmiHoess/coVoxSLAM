#pragma once
#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
int cuda_scan_counts(const std::uint32_t* counts, std::uint64_t* offsets,
                     std::uint64_t* block_sums, std::size_t count, void* stream) noexcept;

struct cuda_tile_coordinate
{
    std::int32_t x{}, y{}, z{};
};

struct cuda_tsdf_voxel {
    float distance{}, weight{};
     std::uint32_t sample_count{}, flags{};
};

struct cuda_tsdf_tile
{
    cuda_tsdf_voxel voxels[512]{};
};

enum class cuda_esdf_unobserved : std::uint32_t { unknown, free, occupied };

struct cuda_site_esdf_parameters
{
    float voxel_size{0.20F};
    float truncation_distance{0.60F};
     float site_band_m{0.20F};
    float minimum_weight{1.0F};
    float maximum_distance_m{8.0F};
    cuda_esdf_unobserved unobserved{cuda_esdf_unobserved::unknown};
    bool include_negative_truncation_band{};
};

struct cuda_site_esdf_voxel {
    float squared_distance_voxels{};
    std::int32_t parent[3]{};
     std::uint32_t flags{};
}
;

inline constexpr std::uint32_t cuda_esdf_observed = 1;
inline constexpr std::uint32_t cuda_esdf_inside = 2;
inline constexpr std::uint32_t cuda_esdf_site = 4;
inline constexpr std::uint32_t cuda_esdf_parent_assigned = 8;

struct cuda_projective_pose
{
    float translation[3]{};
    float rotation_xyzw[4]{0, 0, 0, 1};
};
}
