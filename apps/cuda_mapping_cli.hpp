#pragma once

#include <covoxslam/cuda.hpp>
#include "mapping_cli.hpp"

#include <limits>

namespace covoxslam
{
struct cuda_mapping_cli_options
{
    bool enabled{};
    bool projective{};
    bool ray_capacity_set{};
    bool contribution_capacity_set{};
    cuda_mapper_options mapper;
};

inline bool parse_cuda_mapping_option(std::string_view name, std::string_view value,
    mapping_config& config, bool& enabled, cuda_mapping_cli_options& options)
{
    if (name == "--backend") {
        if (value != "cpu-ray" && value != "cuda-ray" && value != "cuda-projective")
            throw std::invalid_argument("unsupported mapping backend");
        enabled = true;
        options.enabled = value == "cuda-ray";
        options.projective = value == "cuda-projective";
        return true;
    }
    if (name != "--cuda-device" && name != "--cuda-ray-capacity" &&
        name != "--cuda-contribution-capacity" && name != "--cuda-device-bytes" && name != "--maximum-tiles")
        return false;
    std::size_t count{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("expected an unsigned CUDA capacity or device index");
    if (name == "--cuda-device")
    {
        if (count > std::numeric_limits<int>::max()) throw std::invalid_argument("CUDA device index out of range");
        options.mapper.device = static_cast<int>(count);
    }
    else if (name == "--cuda-ray-capacity") {
        options.mapper.ray_capacity = count;
        options.ray_capacity_set = true;
    }
    else if (name == "--cuda-contribution-capacity") {
        options.mapper.contribution_capacity = count;
        options.contribution_capacity_set = true;
    }
     else if (name == "--cuda-device-bytes") options.mapper.maximum_device_bytes = count;
    else config.maximum_tiles = count;
    return true;
}
}
