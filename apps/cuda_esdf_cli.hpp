#pragma once

#include <covoxslam/cuda.hpp>
#include <stdexcept>
#include <string_view>

namespace covoxslam
{
struct cuda_blocks_cli_options {
    cuda_blocks_options options;
    bool specified{};
};

inline bool parse_cuda_esdf_option(std::string_view name, std::string_view value, cuda_blocks_cli_options& config)
{
    if (name == "--esdf-executor") {
        if (value == "ordered") config.options.executor = cuda_blocks_executor::ordered;
        else if (value == "frontier") config.options.executor = cuda_blocks_executor::frontier;
        else throw std::invalid_argument("unsupported CUDA blocks executor");
    }
    else if (name == "--esdf-distance") {
        if (value == "full-euclidean") config.options.distance = cuda_blocks_distance::full_euclidean;
         else if (value == "quasi-euclidean") config.options.distance = cuda_blocks_distance::quasi_euclidean;
        else throw std::invalid_argument("unsupported CUDA blocks distance mode");
    }
    else return false;
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
