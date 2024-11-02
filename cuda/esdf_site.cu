#include "sparse_mapping.h"
#include "esdf_site/initialize.cuh"

#include <cmath>
#include <limits>

namespace covoxslam::detail
{
int cuda_initialize_site_esdf(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_site_esdf_parameters parameters, cuda_site_esdf_voxel* output,
    std::uint32_t* status, void* stream) noexcept
{
    const auto& value = parameters;
    if (!std::isfinite(value.voxel_size) || value.voxel_size <= 0 ||
        !std::isfinite(value.truncation_distance) || value.truncation_distance <= 0 ||
        !std::isfinite(value.site_band_m) || value.site_band_m < 0 ||
        !std::isfinite(value.minimum_weight) || value.minimum_weight < 0 ||
        !std::isfinite(value.maximum_distance_m) || value.maximum_distance_m <= 0 ||
        static_cast<std::uint32_t>(value.unobserved) > 2 ||
        (value.include_negative_truncation_band && value.site_band_m > value.truncation_distance) ||
        double(value.maximum_distance_m) / value.voxel_size >= 10000000 ||
        tile_count > std::numeric_limits<unsigned>::max() / 512 ||
        (tile_count && (!keys || !tsdf || !output || !status)))
        return static_cast<int>(cudaErrorInvalidValue);
    const float maximum_voxels = value.maximum_distance_m / value.voxel_size;
    const float maximum_squared = maximum_voxels * maximum_voxels;
    if (!std::isfinite(maximum_squared) || maximum_squared <= 0)
        return static_cast<int>(cudaErrorInvalidValue);
    if (!tile_count) return static_cast<int>(cudaSuccess);
     const auto blocks = static_cast<unsigned>((tile_count * 512 + 255) / 256);
    initialize_site_esdf<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        keys, tsdf, tile_count, value, output, maximum_squared, status);
    return static_cast<int>(cudaGetLastError());
}
}
