#include <covoxslam/mapping.hpp>
#include <cmath>
#include <limits>

namespace covoxslam
{
bool mapping_config::valid() const noexcept
{
    if (esdf_generator != esdf_backend::cuda_block_band_soa) return false;
    for (const auto value : {voxel_size, truncation_distance, extraction_distance,
                            maximum_weight, minimum_ray_length_m, maximum_ray_length_m,
                            esdf_site_band_m, esdf_maximum_distance_m, esdf_minimum_weight})
        if (!std::isfinite(value) || value <= 0) return false;
    return maximum_tiles > 0 && minimum_ray_length_m <= maximum_ray_length_m &&
        extraction_distance <= truncation_distance &&
        esdf_site_band_m <= truncation_distance && esdf_site_band_m <= esdf_maximum_distance_m &&
        double(esdf_maximum_distance_m) / voxel_size < 10000000 &&
        double(maximum_ray_length_m) + truncation_distance <= std::numeric_limits<float>::max() &&
        (double(maximum_ray_length_m) + truncation_distance) / voxel_size < 10000000;
}

}

namespace covoxslam
{

const char* esdf_backend_name(esdf_backend value) noexcept
{
    switch (value) {
        case esdf_backend::none: return "none";
        case esdf_backend::cpu_blocks_batch: return "cpu-blocks-batch";
        case esdf_backend::cuda_projective_block_band: return "cuda-projective-block-band";
        case esdf_backend::cuda_blocks_batch: return "cuda-blocks-batch";
        case esdf_backend::cuda_block_band_soa: return "cuda-block-band-soa";
    }
    return "unknown";
}
}
