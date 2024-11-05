#pragma once

#include <covoxslam/cuda.hpp>

namespace covoxslam
{
inline std::function<void(submap&)> cuda_esdf_finalizer(const mapping_config& config, int device)
{
    if (config.esdf_generator != esdf_backend::cuda_nvblox_block_band) return {};
    query_cuda_device(device);
    return [config, device](submap& completed)
    {
        cuda_site_esdf_options options;
         options.device = device;
        const auto statistics = build_cuda_site_esdf(completed.tsdf, completed.esdf, config, options);
        completed.config = config;
        completed.statistics.esdf_ms = statistics.elapsed_ms;
        completed.statistics.cuda_uploaded_bytes += statistics.uploaded_bytes;
        completed.statistics.cuda_downloaded_bytes += statistics.downloaded_bytes;
    };
}
}
