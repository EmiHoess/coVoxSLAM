#pragma once
#include <covoxslam/core/byte_units.hpp>
#include <covoxslam/mapping/config.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
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
}

namespace covoxslam::store
{
inline constexpr std::uint32_t default_band_esdf_maximum_rounds = 10'000;
inline constexpr std::size_t default_band_esdf_scratch_bytes = 256U * bytes_per_mebibyte;

struct band_esdf_options
{
    detail::cuda_site_esdf_parameters field{};
    std::uint32_t maximum_rounds{default_band_esdf_maximum_rounds};
     std::size_t scratch_bytes{default_band_esdf_scratch_bytes};

    bool valid() const noexcept
    {
        const auto& p = field;
        const double extent = double(p.maximum_distance_m) / p.voxel_size;
        return maximum_rounds && scratch_bytes &&
            std::isfinite(p.voxel_size) && p.voxel_size > 0 &&
            std::isfinite(p.truncation_distance) && p.truncation_distance > 0 &&
            std::isfinite(p.minimum_weight) && p.minimum_weight >= 0 &&
            std::isfinite(p.site_band_m) && p.site_band_m >= 0 &&
            std::isfinite(p.maximum_distance_m) && p.maximum_distance_m > 0 &&
            std::uint32_t(p.unobserved) <= 2 &&
            (!p.include_negative_truncation_band || p.site_band_m <= p.truncation_distance) &&
            std::isfinite(extent) && extent < maximum_mapping_extent_voxels &&
            float(extent * extent) > 0;
    }
};
inline band_esdf_options band_esdf_options_from_mapping(
    const mapping_config& mapping, bool include_negative_truncation_band) noexcept
{
    band_esdf_options result;
    result.field.voxel_size = mapping.voxel_size;
    result.field.truncation_distance = mapping.truncation_distance;
    result.field.site_band_m = mapping.esdf_site_band_m;
    result.field.minimum_weight = mapping.esdf_minimum_weight;
    result.field.maximum_distance_m = mapping.esdf_maximum_distance_m;
    result.field.include_negative_truncation_band = include_negative_truncation_band;
    return result;
}
struct band_esdf_metrics {
    std::uint64_t tiles{}, scratch_bytes{}, upload_bytes{}, download_bytes{}, peak_device_bytes{};
    std::uint32_t rounds{};
    double allocation_ms{}, upload_ms{}, propagation_ms{}, publication_ms{}, readback_ms{};
    double gpu_ms{}, wait_ms{}, total_ms{};
};
}
