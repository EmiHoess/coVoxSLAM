#pragma once

#include "cuda_mapping_cli.hpp"
#include "cli_lookup.hpp"

namespace covoxslam
{
enum class ray_fusion_option { unknown, mode, threshold };

inline constexpr cli_option_entry<ray_fusion_option> ray_fusion_options[]{
    {"--ray-fusion", ray_fusion_option::mode},
     {"--ray-fusion-short-segment-threshold", ray_fusion_option::threshold}
};
static_assert(valid_cli_options(ray_fusion_options));

struct ray_fusion_cli_options {
    cuda_ray_fusion_mode mode{cuda_ray_fusion_mode::serial};
    std::uint32_t short_segment_threshold{32};
    bool mode_set{};
     bool threshold_set{};
};

inline operation_result<bool> parse_ray_fusion_option(std::string_view name, std::string_view value,
    ray_fusion_cli_options& selection)
{
    switch (lookup_cli_option(name, ray_fusion_options))
    {
    case ray_fusion_option::mode:
        if (value == "serial") selection.mode = cuda_ray_fusion_mode::serial;
        else if (value == "weighted-sum") selection.mode = cuda_ray_fusion_mode::weighted_sum;
        else if (value == "weighted-sum-hybrid") selection.mode = cuda_ray_fusion_mode::weighted_sum_hybrid;
        else if (value == "capped-affine") selection.mode = cuda_ray_fusion_mode::capped_affine;
         else if (value == "capped-affine-hybrid") selection.mode = cuda_ray_fusion_mode::capped_affine_hybrid;
        else return std::unexpected(operation_error{"unsupported ray fusion policy"});
        selection.mode_set = true;
        break;
    case ray_fusion_option::threshold:
    {
        std::uint32_t threshold{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), threshold);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !threshold)
            return std::unexpected(operation_error{"expected a positive ray fusion threshold"});
        selection.short_segment_threshold = threshold;
         selection.threshold_set = true;
        break;
    }
    case ray_fusion_option::unknown:
        return false;
    }
    return true;
}

inline operation_status validate_ray_fusion_selection(bool enabled, bool mapping_enabled,
    ray_fusion_cli_options selection)
{
    const bool hybrid = selection.mode == cuda_ray_fusion_mode::weighted_sum_hybrid ||
        selection.mode == cuda_ray_fusion_mode::capped_affine_hybrid;
    if ((selection.mode_set || selection.threshold_set) && !mapping_enabled)
        return std::unexpected(operation_error{"ray fusion options require a mapping backend"});
    if (!enabled && (selection.mode != cuda_ray_fusion_mode::serial || selection.threshold_set))
        return std::unexpected(operation_error{"nonserial ray fusion requires --backend cuda-ray"});
     if (selection.threshold_set && !hybrid)
        return std::unexpected(operation_error{"ray fusion threshold requires a hybrid reducer"});
    return {};
}
}
