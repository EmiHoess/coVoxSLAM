#pragma once

#include <covoxslam/mapping.hpp>
#include <covoxslam/result.hpp>
#include "cli_lookup.hpp"

#include <charconv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace covoxslam
{
enum class mapping_option
{
    unknown, backend, esdf_generator, esdf_maximum_distance, esdf_site_band,
     esdf_minimum_weight, voxel_size, truncation_distance, minimum_ray_length, maximum_ray_length, maximum_weight
};
inline constexpr cli_option_entry<mapping_option> mapping_options[]{
    {"--backend", mapping_option::backend}, {"--esdf-generator", mapping_option::esdf_generator},
    {"--esdf-maximum-distance", mapping_option::esdf_maximum_distance},
     {"--esdf-site-band", mapping_option::esdf_site_band}, {"--esdf-minimum-weight", mapping_option::esdf_minimum_weight},
    {"--voxel-size", mapping_option::voxel_size}, {"--truncation-distance", mapping_option::truncation_distance},
    {"--minimum-ray-length", mapping_option::minimum_ray_length},
    {"--maximum-ray-length", mapping_option::maximum_ray_length}, {"--maximum-weight", mapping_option::maximum_weight}
};

enum class esdf_value { unknown, none, cpu_blocks, projective_band, blocks_batch };
inline constexpr cli_option_entry<esdf_value> esdf_values[]{
    {"none", esdf_value::none}, {"cpu-blocks-batch", esdf_value::cpu_blocks},
     {"cuda-projective-block-band", esdf_value::projective_band}, {"cuda-blocks-batch", esdf_value::blocks_batch}
};

inline float mapping_scalar(std::string_view text)
{
    float result{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
     if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !std::isfinite(result))
        throw std::invalid_argument("expected a finite mapping scalar");
    return result;
}

inline operation_result<bool> parse_mapping_option(std::string_view name, std::string_view value,
                                mapping_config& config, bool& enabled)
{
    switch (lookup_cli_option(name, mapping_options))
    {
    case mapping_option::backend:

        if (value != "cpu-ray") return std::unexpected(operation_error{"unsupported mapping backend"});
        enabled = true;
        return true;
    case mapping_option::esdf_generator:
        switch (lookup_cli_option(value, esdf_values))
        {
        case esdf_value::none: config.esdf_generator = esdf_backend::none; break;
        case esdf_value::cpu_blocks: config.esdf_generator = esdf_backend::cpu_blocks_batch; break;
        case esdf_value::projective_band: config.esdf_generator = esdf_backend::cuda_projective_block_band; break;
         case esdf_value::blocks_batch: config.esdf_generator = esdf_backend::cuda_blocks_batch; break;
        case esdf_value::unknown: return std::unexpected(operation_error{"unsupported ESDF generator"});
        }
        return true;
    default: break;
    }
    if (name == "--esdf-maximum-distance") config.esdf_maximum_distance_m = mapping_scalar(value);
    else if (name == "--esdf-site-band") config.esdf_site_band_m = mapping_scalar(value);
    else if (name == "--esdf-minimum-weight") config.esdf_minimum_weight = mapping_scalar(value);
    else if (name == "--voxel-size") config.voxel_size = mapping_scalar(value);
    else if (name == "--truncation-distance") config.truncation_distance = mapping_scalar(value);
    else if (name == "--minimum-ray-length") config.minimum_ray_length_m = mapping_scalar(value);
     else if (name == "--maximum-ray-length") config.maximum_ray_length_m = mapping_scalar(value);
    else if (name == "--maximum-weight") config.maximum_weight = mapping_scalar(value);
    else return false;
    return true;
}

template<class mapper_type>
inline pipeline_callbacks mapping_callbacks(mapper_type& mapper,
    std::function<void(const submap&)> publish = {}, std::function<void(submap&)> finalize = {})
{
    pipeline_callbacks callbacks;
    callbacks.begin = [&mapper](const submap_frame& frame) { mapper.begin(frame); };
    callbacks.integrate_scan = [&mapper](const submap_frame&, const frame_descriptor&,
                                         std::span<const ray> rays, const pose& reference)
    {
        if constexpr (requires { mapper.integrate_scan(rays, reference); }) mapper.integrate_scan(rays, reference);
        else mapper.integrate_scan(rays);
    };
    callbacks.finish = [&mapper, publish, finalize](const submap_frame& frame)
      {
        auto completed = mapper.finish(frame);
        if (finalize) finalize(completed);
        if (publish) publish(completed);
        const auto& statistics = completed.statistics;
        std::cout << "tsdf " << frame.id << ": " << completed.tsdf.size() << " tiles, "
                  << statistics.rays_integrated << " integrated, " << statistics.rays_rejected
                  << " rejected, " << statistics.voxel_updates << " updates, "
                   << statistics.integration_ms << " ms\n";
        if (statistics.cuda_uploaded_bytes || statistics.cuda_downloaded_bytes)
            std::cout << "cuda " << statistics.cuda_contributing_rays << " contributing rays, "
                      << statistics.cuda_voxel_segments << " voxel segments, " << statistics.cuda_uploaded_bytes
                      << " uploaded bytes, " << statistics.cuda_downloaded_bytes << " downloaded bytes\n";
        std::cout << "esdf " << frame.id << ": " << esdf_backend_name(completed.config.esdf_generator)
                  << ", " << completed.esdf.size() << " tiles, " << statistics.esdf_ms << " ms\n";
    };
    return callbacks;
}
}
