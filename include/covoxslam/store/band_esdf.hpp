#pragma once
#include "../../../cuda/sparse_mapping.h"
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace covoxslam::store
{
struct band_esdf_options
{
    detail::cuda_site_esdf_parameters field{};
    std::uint32_t maximum_rounds{10000};
     std::size_t scratch_bytes{256U * 1024U * 1024U};

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
            std::isfinite(extent) && extent < 10000000 &&
            float(extent * extent) > 0;
    }
};
struct band_esdf_metrics {
    std::uint64_t tiles{}, scratch_bytes{}, upload_bytes{}, download_bytes{}, peak_device_bytes{};
    std::uint32_t rounds{};
    double allocation_ms{}, upload_ms{}, propagation_ms{}, publication_ms{}, readback_ms{};
    double gpu_ms{}, wait_ms{}, total_ms{};
};
}
