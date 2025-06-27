#pragma once
#include "cli_lookup.hpp"
#include <covoxslam/result.hpp>
#include <string>

namespace covoxslam
{
enum class retired_option { unknown, unavailable };
inline constexpr cli_option_entry<retired_option> retired_options[]{
    {"--cuda-contribution-capacity", retired_option::unavailable},
    {"--cuda-ray-capacity", retired_option::unavailable},
    {"--free-space-min-weight", retired_option::unavailable},
    {"--free-space-refresh-scans", retired_option::unavailable},
    {"--projective-allow-clear", retired_option::unavailable},
    {"--projective-candidate-capacity", retired_option::unavailable},
    {"--projective-constant-weight", retired_option::unavailable},
    {"--projective-depth-discontinuity", retired_option::unavailable},
    {"--projective-interpolation", retired_option::unavailable},
    {"--projective-nearest-ray-distance", retired_option::unavailable},
    {"--projective-raycast-subsampling-factor", retired_option::unavailable},
    {"--projective-sparsity-compensation", retired_option::unavailable},
    {"--projective-sparsity-factor", retired_option::unavailable},
    {"--projective-voxel-carving", retired_option::unavailable},
    {"--projective-weight-dropoff", retired_option::unavailable},
    {"--ray-fusion", retired_option::unavailable},
    {"--ray-fusion-short-segment-threshold", retired_option::unavailable},
    {"--ray-segment-length", retired_option::unavailable},
    {"--resident-contributions", retired_option::unavailable},
    {"--resident-rays", retired_option::unavailable},
    {"--resident-tiles", retired_option::unavailable},
    {"--warp-cache-entries", retired_option::unavailable},
    {"--warp-count", retired_option::unavailable},
    {"--warp-ray-order", retired_option::unavailable},
    {"--sensor-grid-candidate-capacity", retired_option::unavailable},
    {"--sensor-grid-backend", retired_option::unavailable},
    {"--sensor-grid-factor", retired_option::unavailable},
    {"--sensor-grid-first-entry", retired_option::unavailable},
    {"--sensor-grid-frames", retired_option::unavailable},
    {"--sensor-grid-radius", retired_option::unavailable},
    {"--sensor-grid-ray-cap", retired_option::unavailable},
    {"--sensor-grid-truncation", retired_option::unavailable},
    {"--sensor-grid-voxel-size", retired_option::unavailable}
};

inline operation_status validate_active_option(std::string_view name)
{
    switch (lookup_cli_option(name, retired_options)) {
    case retired_option::unavailable:
        return std::unexpected(operation_error{std::string(name) + " is not applicable to coVoxSLAM"});
    default: return {};
    }
}
}
