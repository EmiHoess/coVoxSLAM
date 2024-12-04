#pragma once

#include <covoxslam/result.hpp>
#include <cstdint>
#include <span>
#include <vector>

namespace covoxslam::sensor_grid
{
struct direction { float x{}, y{}, z{}; };
struct config
{
    float voxel_size{0.2F}, radius{16.0F}, truncation{0.6F};
     float minimum_range{0.1F}, maximum_weight{10000.0F};
    float sparsity_compensation{20.0F};
    bool weight_dropoff{true};
};
struct intersection { std::uint32_t ray{}; float entry_t{}, projected{}; };
static_assert(sizeof(intersection) == 12);

struct geometry
{
    std::uint32_t half_side{}, side{}, voxels{};
    float origin{};
};
struct lookup
{
    config settings{};
    geometry grid{};
    std::uint32_t ray_count{}, selected_rays{}, reduction_factor{1};
     std::vector<std::uint32_t> offsets, active_voxels;
    std::vector<intersection> intersections;
};
operation_result<geometry> make_geometry(const config& settings);
operation_result<lookup> build_lookup(const config& settings,
    std::span<const direction> directions, std::uint32_t reduction_factor = 1);
operation_status validate_lookup(const lookup& table);
}
