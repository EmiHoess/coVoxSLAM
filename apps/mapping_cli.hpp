#pragma once

#include <covoxslam/mapping.hpp>

#include <charconv>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace covoxslam
{
inline float mapping_scalar(std::string_view text)
{
    float result{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
     if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !std::isfinite(result))
        throw std::invalid_argument("expected a finite mapping scalar");
    return result;
}

inline bool parse_mapping_option(std::string_view name, std::string_view value,
                                mapping_config& config, bool& enabled)
{
    if (name == "--backend") {
        if (value != "cpu-ray") throw std::invalid_argument("unsupported mapping backend");
        enabled = true;
    }
    else if (name == "--voxel-size") config.voxel_size = mapping_scalar(value);
    else if (name == "--truncation-distance") config.truncation_distance = mapping_scalar(value);
    else if (name == "--minimum-ray-length") config.minimum_ray_length_m = mapping_scalar(value);
     else if (name == "--maximum-ray-length") config.maximum_ray_length_m = mapping_scalar(value);
    else if (name == "--maximum-weight") config.maximum_weight = mapping_scalar(value);
    else return false;
    return true;
}

inline pipeline_callbacks cpu_mapping_callbacks(cpu_mapper& mapper)
{
    pipeline_callbacks callbacks;
    callbacks.begin = [&mapper](const submap_frame& frame) { mapper.begin(frame); };
    callbacks.integrate_scan = [&mapper](const submap_frame&, const frame_descriptor&,
                                         std::span<const ray> rays, const pose&)
    {
        mapper.integrate_scan(rays);
    };
    callbacks.finish = [&mapper](const submap_frame& frame)
      {
        const auto completed = mapper.finish(frame);
        const auto& statistics = completed.statistics;
        std::cout << "tsdf " << frame.id << ": " << completed.tsdf.size() << " tiles, "
                  << statistics.rays_integrated << " integrated, " << statistics.rays_rejected
                  << " rejected, " << statistics.voxel_updates << " updates, "
                   << statistics.integration_ms << " ms\n";
    };
    return callbacks;
}
}
