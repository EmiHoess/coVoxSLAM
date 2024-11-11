#pragma once

#include "cuda_mapping_cli.hpp"

namespace covoxslam
{
inline bool parse_projective_option(std::string_view name, std::string_view value, cuda_projective_options& options, bool& selected)
{
    if (name == "--projective-interpolation") {
        if (value == "nearest") options.interpolation = cuda_projective_interpolation::nearest;
        else if (value == "linear-nearest") options.interpolation = cuda_projective_interpolation::linear_nearest;
        else throw std::invalid_argument("unsupported projective interpolation");
    }
    else if (name == "--projective-depth-discontinuity") options.depth_discontinuity_voxels = mapping_scalar(value);
    else if (name == "--projective-nearest-ray-distance") options.nearest_ray_distance_voxels = mapping_scalar(value);
    else if (name == "--projective-sparsity-factor") options.sparsity_compensation_factor = mapping_scalar(value);
    else if (name == "--projective-candidate-capacity") {
        std::size_t capacity{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), capacity);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
            throw std::invalid_argument("expected projective candidate capacity");
        options.candidate_tile_capacity = capacity;
    }
    else {
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
