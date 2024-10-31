#pragma once

#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
inline constexpr std::uint32_t cuda_mapping_abi_version = 1;

struct cuda_device_ray {
    float origin[3]{};
    float endpoint[3]{};
     float weight{1.0F};
    std::uint64_t stamp_ns{};
    std::uint32_t frame_id{};
    std::uint32_t column{};
    std::uint32_t ring{};
    std::uint32_t ordinal{};
};

int cuda_scan_counts(const std::uint32_t* counts, std::uint64_t* offsets,
                     std::uint64_t* block_sums, std::size_t count, void* stream) noexcept;
}

namespace covoxslam::detail
{
struct cuda_ray_parameters
{
    float voxel_size{0.20F};
    float truncation_distance{0.60F};
     float minimum_ray_length_m{0.10F};
    float maximum_ray_length_m{16.0F};
    bool voxel_carving{true};
     bool allow_clear{};
};

struct cuda_ray_contribution {
    std::int32_t voxel[3]{};
    float distance{};
    float weight{};
     std::uint32_t ray_ordinal{};
    std::uint32_t traversal_ordinal{};
};

int cuda_count_ray_contributions(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, std::uint32_t* counts, std::uint32_t* status, void* stream) noexcept;
int cuda_emit_ray_contributions(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, const std::uint32_t* counts, const std::uint64_t* offsets,
    cuda_ray_contribution* output, std::uint64_t capacity, std::uint32_t* status, void* stream) noexcept;
}

namespace covoxslam::detail
{
struct cuda_ray_summary
{
    std::uint64_t contributions{};
     std::uint32_t contributing_rays{};
    std::uint32_t status{};
};

struct cuda_ray_device_view {
    cuda_ray_contribution* contributions{};
    std::uint64_t count{};
    std::uint64_t capacity{};
     std::uint32_t* counts{};
    std::uint64_t* offsets{};
    std::uint64_t* block_sums{};
    void* stream{};
    int device{};
    std::size_t ray_capacity{};
};
}

namespace covoxslam::detail
{
int cuda_summarize_ray_contributions(const std::uint32_t* counts, const std::uint64_t* offsets,
    std::size_t count, cuda_ray_summary* summary, void* stream) noexcept;
}

namespace covoxslam::detail
{
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

struct cuda_voxel_segment {
    std::int32_t voxel[3]{};
    std::uint32_t count{};
     std::uint64_t begin{};
    std::uint32_t tile_slot{UINT32_MAX};
    std::uint32_t local_index{};
};

struct cuda_fusion_parameters
{
    float truncation_distance{0.60F};
    float maximum_weight{10000.0F};
     float extraction_distance{0.20F};
};

static_assert(sizeof(cuda_tile_coordinate) == 12 && sizeof(cuda_tsdf_voxel) == 16);
static_assert(sizeof(cuda_tsdf_tile) == 8192 && sizeof(cuda_voxel_segment) == 32);

int cuda_stable_sort_contributions(cuda_ray_contribution* values, cuda_ray_contribution* scratch,
    std::uint32_t* flags, std::uint64_t* offsets, std::uint64_t* block_sums,
    std::size_t count, void* stream) noexcept;
int cuda_build_voxel_segments(const cuda_ray_contribution* values, std::size_t count,
    std::uint32_t* flags, std::uint64_t* offsets, std::uint64_t* block_sums,
    cuda_voxel_segment* segments, std::uint32_t capacity, std::uint32_t* segment_count,
    std::uint32_t* status, void* stream) noexcept;
int cuda_fuse_voxel_segments(const cuda_ray_contribution* values, std::size_t count,
    const cuda_voxel_segment* segments, std::uint32_t segment_count,
    const cuda_tile_coordinate* keys, cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    cuda_fusion_parameters config, std::uint32_t* status, void* stream) noexcept;
}
