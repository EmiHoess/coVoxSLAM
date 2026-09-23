#pragma once

#include <covoxslam/core/byte_units.hpp>
#include <cstddef>
#include <type_traits>

namespace covoxslam
{
enum class esdf_backend {
    none = 0,
    cpu_blocks_batch = 1,
    cuda_projective_block_band = 2,
    cuda_blocks_batch = 3,
    cuda_block_band_soa = 4
};

const char* esdf_backend_name(esdf_backend value) noexcept;
inline constexpr std::size_t default_mapping_device_bytes = 2U * bytes_per_gibibyte;
inline constexpr std::size_t maximum_mapping_tiles = 1U << 20U;

struct mapping_config
{
    float voxel_size{0.20F};
    float truncation_distance{0.60F};
    float extraction_distance{0.20F};
     float maximum_weight{10000.0F};
    float minimum_ray_length_m{0.10F};
    float maximum_ray_length_m{16.0F};
    std::size_t maximum_tiles{32768};
    float esdf_site_band_m{0.20F};
    float esdf_maximum_distance_m{8.0F};
     float esdf_minimum_weight{1.0F};
    esdf_backend esdf_generator{esdf_backend::cuda_block_band_soa};
    bool valid() const noexcept;
};

static_assert(static_cast<int>(esdf_backend::none) == 0);
static_assert(static_cast<int>(esdf_backend::cpu_blocks_batch) == 1);
static_assert(static_cast<int>(esdf_backend::cuda_projective_block_band) == 2);
static_assert(static_cast<int>(esdf_backend::cuda_blocks_batch) == 3);
static_assert(static_cast<int>(esdf_backend::cuda_block_band_soa) == 4);
static_assert(std::is_same_v<std::underlying_type_t<esdf_backend>, int>);

static_assert(std::is_aggregate_v<mapping_config>);
static_assert(std::is_standard_layout_v<mapping_config>);
static_assert(std::is_trivially_copyable_v<mapping_config>);
static_assert(sizeof(mapping_config) == 48);
static_assert(alignof(mapping_config) == alignof(std::size_t));
static_assert(offsetof(mapping_config, voxel_size) == 0);
static_assert(offsetof(mapping_config, maximum_tiles) == 24);
static_assert(offsetof(mapping_config, esdf_site_band_m) == 32);
static_assert(offsetof(mapping_config, esdf_generator) == 44);
}
