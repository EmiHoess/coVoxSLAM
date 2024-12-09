#pragma once

#include <cstdint>
#include <type_traits>

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
struct geometry
{
    std::uint32_t half_side{}, side{}, voxels{};
    float origin{};
};

struct voxel
{
    float distance{}, weight{};
     std::uint32_t contributions{}, candidates_tested{};
};
struct gather_policy { std::uint32_t max_rays{}; bool first_entry_only{}; };
struct lookup_view
{
    const std::uint32_t* offsets{};
    const intersection* intersections{};
    const std::uint32_t* active_voxels{};
    std::uint32_t active_count{}, ray_count{}, entry_count{}, voxel_count{};
    config settings{};
};
struct field_view
{
    const voxel* voxels{};
    std::uint32_t side{};
    float voxel_size{}, origin{};
};
}
