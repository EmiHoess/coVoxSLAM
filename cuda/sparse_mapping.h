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
