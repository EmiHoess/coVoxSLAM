#pragma once

#include "cuda_mapping_cli.hpp"

namespace covoxslam
{
enum class projective_option
{
    unknown, interpolation, depth_discontinuity, nearest_ray_distance, sparsity_factor,
     candidate_capacity, constant_weight, weight_dropoff, sparsity_compensation, voxel_carving, allow_clear
};
inline constexpr cli_option_entry<projective_option> projective_options[]{
    {"--projective-interpolation", projective_option::interpolation},
    {"--projective-depth-discontinuity", projective_option::depth_discontinuity},
    {"--projective-nearest-ray-distance", projective_option::nearest_ray_distance},
    {"--projective-sparsity-factor", projective_option::sparsity_factor},
    {"--projective-candidate-capacity", projective_option::candidate_capacity},
    {"--projective-constant-weight", projective_option::constant_weight},
     {"--projective-weight-dropoff", projective_option::weight_dropoff},
    {"--projective-sparsity-compensation", projective_option::sparsity_compensation},
    {"--projective-voxel-carving", projective_option::voxel_carving},
    {"--projective-allow-clear", projective_option::allow_clear}
};
static_assert(valid_cli_options(projective_options));

enum class projective_interpolation_value { unknown, nearest, linear_nearest };
inline constexpr cli_option_entry<projective_interpolation_value> projective_interpolation_values[]{
    {"nearest", projective_interpolation_value::nearest}, {"linear-nearest", projective_interpolation_value::linear_nearest}
};
static_assert(valid_cli_options(projective_interpolation_values));

inline operation_result<bool> parse_projective_option(std::string_view name, std::string_view value, cuda_projective_options& options, bool& selected)
{
    float* scalar{};
    switch (lookup_cli_option(name, projective_options))
    {
    case projective_option::interpolation:
        switch (lookup_cli_option(value, projective_interpolation_values)) {
        case projective_interpolation_value::nearest: options.interpolation = cuda_projective_interpolation::nearest; break;
         case projective_interpolation_value::linear_nearest: options.interpolation = cuda_projective_interpolation::linear_nearest; break;
        default: return std::unexpected(operation_error{"unsupported projective interpolation"});
        }
        selected = true;
        return true;
    case projective_option::depth_discontinuity: scalar = &options.depth_discontinuity_voxels; break;
     case projective_option::nearest_ray_distance: scalar = &options.nearest_ray_distance_voxels; break;
    case projective_option::sparsity_factor: scalar = &options.sparsity_compensation_factor; break;
    case projective_option::candidate_capacity: {
        std::size_t capacity{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), capacity);

        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
            return std::unexpected(operation_error{"expected projective candidate capacity"});
        options.candidate_tile_capacity = capacity;
        selected = true;
        return true;
    }
    default: break;
    }

    if (scalar) {
        float number{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);

        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !std::isfinite(number))
            return std::unexpected(operation_error{"expected a finite mapping scalar"});
        *scalar = number;
         selected = true;
        return true;
    }
    {
        bool* selected{};
        if (name == "--projective-constant-weight") selected = &options.constant_weight;
        else if (name == "--projective-weight-dropoff") selected = &options.weight_dropoff;
        else if (name == "--projective-sparsity-compensation") selected = &options.sparsity_compensation;
        else if (name == "--projective-voxel-carving") selected = &options.voxel_carving;
        else if (name == "--projective-allow-clear") selected = &options.allow_clear;
        else return false;
        if (value != "true" && value != "false") throw std::invalid_argument("expected true or false");
        *selected = value == "true";
    }
    selected = true;
    return true;
}
}
