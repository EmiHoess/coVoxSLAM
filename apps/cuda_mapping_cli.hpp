#pragma once

#include <covoxslam/cuda.hpp>
#include "mapping_cli.hpp"

#include <limits>

namespace covoxslam
{
enum class cuda_mapping_option
{
    unknown, backend, device, ray_capacity, contribution_capacity, device_bytes, maximum_tiles
};
inline constexpr cli_option_entry<cuda_mapping_option> cuda_mapping_options[]{
    {"--backend", cuda_mapping_option::backend}, {"--cuda-device", cuda_mapping_option::device},
     {"--cuda-ray-capacity", cuda_mapping_option::ray_capacity},
    {"--cuda-contribution-capacity", cuda_mapping_option::contribution_capacity},
    {"--cuda-device-bytes", cuda_mapping_option::device_bytes}, {"--maximum-tiles", cuda_mapping_option::maximum_tiles}
};
static_assert(valid_cli_options(cuda_mapping_options));

struct cuda_mapping_cli_options
{
    bool enabled{};
    bool projective{};
     bool calibrated_policy{};
    bool ray_capacity_set{};
    bool contribution_capacity_set{};
    cuda_mapper_options mapper;
};

inline operation_result<bool> parse_cuda_mapping_option(std::string_view name, std::string_view value,
    mapping_config& config, bool& enabled, cuda_mapping_cli_options& options)
{
    switch (lookup_cli_option(name, cuda_mapping_options)) {
    case cuda_mapping_option::backend:
        if (value != "cpu-ray" && value != "cuda-ray" && value != "cuda-projective" && value != "cuda-projective-calibrated")
            return std::unexpected(operation_error{"unsupported mapping backend"});
        enabled = true;
        options.enabled = value == "cuda-ray";
        options.calibrated_policy = value == "cuda-projective-calibrated";
        options.projective = value == "cuda-projective" || options.calibrated_policy;
        return true;
    default: break;
    }
    if (name != "--cuda-device" && name != "--cuda-ray-capacity" &&
        name != "--cuda-contribution-capacity" && name != "--cuda-device-bytes" && name != "--maximum-tiles")
        return false;
    std::size_t count{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        return std::unexpected(operation_error{"expected an unsigned CUDA capacity or device index"});
    if (name == "--cuda-device")
    {
        if (count > std::numeric_limits<int>::max()) return std::unexpected(operation_error{"CUDA device index out of range"});
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
