#pragma once

#include <covoxslam/cuda.hpp>
#include "cli_lookup.hpp"
#include <stdexcept>
#include <string_view>

namespace covoxslam
{
struct cuda_blocks_cli_options {
    cuda_blocks_options options;
    bool specified{};
};

enum class esdf_cli_option { unknown, executor, distance };
inline constexpr cli_option_entry<esdf_cli_option> esdf_cli_options[]{
    {"--esdf-executor", esdf_cli_option::executor}, {"--esdf-distance", esdf_cli_option::distance}
};
static_assert(valid_cli_options(esdf_cli_options));
enum class esdf_executor_value { unknown, ordered, frontier };
inline constexpr cli_option_entry<esdf_executor_value> esdf_executor_values[]{
    {"ordered", esdf_executor_value::ordered}, {"frontier", esdf_executor_value::frontier}
};
static_assert(valid_cli_options(esdf_executor_values));
enum class esdf_distance_value { unknown, full, quasi };
inline constexpr cli_option_entry<esdf_distance_value> esdf_distance_values[]{
    {"full-euclidean", esdf_distance_value::full}, {"quasi-euclidean", esdf_distance_value::quasi}
};
static_assert(valid_cli_options(esdf_distance_values));

inline operation_result<bool> parse_cuda_esdf_option(std::string_view name, std::string_view value, cuda_blocks_cli_options& config)
{
    switch (lookup_cli_option(name, esdf_cli_options)) {
    case esdf_cli_option::executor:
        switch (lookup_cli_option(value, esdf_executor_values)) {
        case esdf_executor_value::ordered: config.options.executor = cuda_blocks_executor::ordered; break;
         case esdf_executor_value::frontier: config.options.executor = cuda_blocks_executor::frontier; break;
        default: return std::unexpected(operation_error{"unsupported CUDA blocks executor"});
        }
        break;
    case esdf_cli_option::distance:
        switch (lookup_cli_option(value, esdf_distance_values)) {
        case esdf_distance_value::full: config.options.distance = cuda_blocks_distance::full_euclidean; break;
         case esdf_distance_value::quasi: config.options.distance = cuda_blocks_distance::quasi_euclidean; break;
        default: return std::unexpected(operation_error{"unsupported CUDA blocks distance mode"});
        }
        break;
    default: return false;
    }
    config.specified = true;
    return true;
}

inline std::function<void(submap&)> cuda_esdf_finalizer(const mapping_config& config, int device,
    const cuda_blocks_cli_options& blocks = {})
{
    if (blocks.specified && config.esdf_generator != esdf_backend::cuda_blocks_batch)
        throw std::invalid_argument("blocks execution options require cuda-blocks-batch");
    if (config.esdf_generator != esdf_backend::cuda_projective_block_band &&
        config.esdf_generator != esdf_backend::cuda_blocks_batch) return {};
    query_cuda_device(device);
    return [config, device, blocks](submap& completed)
    {
        const auto record = [&](const auto& statistics) {
            completed.statistics.esdf_ms = statistics.elapsed_ms;
            completed.statistics.cuda_uploaded_bytes += statistics.uploaded_bytes;
            completed.statistics.cuda_downloaded_bytes += statistics.downloaded_bytes;
        };
        if (config.esdf_generator == esdf_backend::cuda_blocks_batch) {
            auto options = blocks.options;
             options.device = device;
            record(build_cuda_blocks_esdf(completed.tsdf, completed.esdf, config, options));
        }
        else
        {
            cuda_site_esdf_options options;
            options.device = device;
            record(build_cuda_site_esdf(completed.tsdf, completed.esdf, config, options));
        }
        completed.config = config;
    };
}
}
