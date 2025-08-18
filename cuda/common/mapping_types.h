#pragma once
#include <covoxslam/voxel/esdf_encodings.hpp>
#include <covoxslam/voxel/records.hpp>
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

using cuda_tsdf_voxel = tsdf_record;

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

struct cuda_projective_pose
{
    float translation[3]{};
    float rotation_xyzw[4]{0, 0, 0, 1};
};
}
