#pragma once
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

struct cuda_projective_pose
{
    float translation[3]{};
    float rotation_xyzw[4]{0, 0, 0, 1};
};
}
