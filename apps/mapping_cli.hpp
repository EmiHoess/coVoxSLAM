#pragma once

#include <covoxslam/core/byte_units.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/mapping/config.hpp>
#include "cli_lookup.hpp"

#include <charconv>
#include <cmath>
#include <iostream>
#include <limits>
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

enum class esdf_value { unknown, block_band_soa };
inline constexpr cli_option_entry<esdf_value> esdf_values[]{
    {"cuda-block-band-soa", esdf_value::block_band_soa}
};

struct mapping_device_options
{
    int device{};
    std::size_t maximum_device_bytes{2U * bytes_per_gibibyte};
};
enum class mapping_device_option { unknown, device, bytes, tiles };
inline constexpr cli_option_entry<mapping_device_option> mapping_device_options_table[]{
    {"--cuda-device", mapping_device_option::device},
    {"--cuda-device-bytes", mapping_device_option::bytes},
    {"--maximum-tiles", mapping_device_option::tiles}
};

inline operation_result<bool> parse_mapping_device(std::string_view name, std::string_view value,
    mapping_config& config, int& device, std::size_t& bytes)
{
    const auto id = lookup_cli_option(name, mapping_device_options_table);

    if (id == mapping_device_option::unknown) return false;
    std::size_t count{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);

    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        return std::unexpected(operation_error{"expected an unsigned CUDA capacity or device index"});
    switch (id) {
    case mapping_device_option::device:
        if (count > std::numeric_limits<int>::max())
            return std::unexpected(operation_error{"CUDA device index out of range"});
        device = static_cast<int>(count);
         return true;
    case mapping_device_option::bytes:
        if (!count) return std::unexpected(operation_error{"CUDA memory budget must be positive"});
        bytes = count;
        return true;
    case mapping_device_option::tiles:
        if (!count || count > 1048576)
            return std::unexpected(operation_error{"mapping tile capacity is out of range"});
        config.maximum_tiles = count;
        return true;
    default: return false;
    }
}

inline operation_result<bool> parse_mapping_option(std::string_view name, std::string_view value,
                                mapping_config& config, bool& enabled)
{
    float* scalar{};
    switch (lookup_cli_option(name, mapping_options))
    {
    case mapping_option::backend:

        if (value != "cuda-sensor-grid-combined")
            return std::unexpected(operation_error{"mapping backend is not applicable to coVoxSLAM"});
        enabled = true;
        return true;
    case mapping_option::esdf_generator:
        switch (lookup_cli_option(value, esdf_values))
        {
        case esdf_value::block_band_soa: config.esdf_generator = esdf_backend::cuda_block_band_soa; break;
        case esdf_value::unknown: return std::unexpected(operation_error{"unsupported ESDF generator"});
        }
        return true;
    case mapping_option::voxel_size: scalar = &config.voxel_size; break;
    case mapping_option::truncation_distance: scalar = &config.truncation_distance; break;
    case mapping_option::minimum_ray_length: scalar = &config.minimum_ray_length_m; break;
     case mapping_option::maximum_ray_length: scalar = &config.maximum_ray_length_m; break;
    case mapping_option::maximum_weight: scalar = &config.maximum_weight; break;
    case mapping_option::esdf_maximum_distance: scalar = &config.esdf_maximum_distance_m; break;
     case mapping_option::esdf_site_band: scalar = &config.esdf_site_band_m; break;
    case mapping_option::esdf_minimum_weight: scalar = &config.esdf_minimum_weight; break;
    case mapping_option::unknown: return false;
    }

    if (scalar)
    {
        float parsed_value{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), parsed_value);

        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !std::isfinite(parsed_value))
            return std::unexpected(operation_error{"expected a finite mapping scalar"});
        *scalar = parsed_value;
        return true;
    }
    return false;
}

}
