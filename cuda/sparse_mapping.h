#pragma once

#include <cstddef>
#include <cstdint>
#include "common/mapping_types.h"

namespace covoxslam::detail
{

int cuda_initialize_site_esdf(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_site_esdf_parameters parameters, cuda_site_esdf_voxel* output,
    std::uint32_t* status, void* stream) noexcept;
int cuda_sweep_site_esdf(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_site_esdf_voxel* output, float maximum_squared, std::uint32_t* changed, void* stream) noexcept;
int cuda_relax_site_esdf_neighbors(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_site_esdf_voxel* output, float maximum_squared, std::uint32_t* changed, void* stream) noexcept;
}

namespace covoxslam::detail
{
struct cuda_blocks_voxel
{
    float distance_m{};
    std::int32_t parent[3]{};
     std::uint32_t flags{};
};

struct cuda_blocks_parameters {
    float voxel_size{0.20F};
    float seed_band_m{0.20F};
    float maximum_distance_m{8.0F};
     float minimum_weight{1.0F};
    float minimum_difference_m{0.0002F};
    std::uint32_t bucket_count{64};
    bool full_euclidean{true};
};

struct cuda_blocks_queue
{
    std::uint32_t size{}, cursor{}, status{}, frontier_count{};
    std::uint64_t pushes{}, pops{}, accepted{}, relaxations{}, eligible{}, seeds{};
};

struct cuda_blocks_proposal {
    std::uint32_t target{UINT32_MAX};
    float source_distance_m{}, step_distance_m{};
     std::int32_t parent[3]{};
};

struct cuda_blocks_workspace
{
    cuda_blocks_voxel* values{};
    std::uint32_t* next{};
    std::uint32_t* queued{};
    std::uint32_t* heads{};
     std::uint32_t* tails{};
    const std::uint32_t* order{};
    std::uint32_t* sources{};
    std::uint32_t* winners{};
    cuda_blocks_proposal* proposals{};
    cuda_blocks_queue* queue{};
};

static_assert(sizeof(cuda_blocks_voxel) == 20 && sizeof(cuda_blocks_queue) == 64);
static_assert(sizeof(cuda_blocks_proposal) == 24);
int cuda_initialize_blocks(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_blocks_parameters config, cuda_blocks_workspace work, void* stream) noexcept;
int cuda_enqueue_blocks_seeds(std::size_t count, cuda_blocks_parameters config,
    cuda_blocks_workspace work, void* stream) noexcept;
int cuda_propagate_blocks_ordered(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_blocks_parameters config, cuda_blocks_workspace work, std::uint32_t maximum_pops, void* stream) noexcept;
int cuda_propagate_blocks_frontier(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_blocks_parameters config, cuda_blocks_workspace work, std::uint32_t capacity, void* stream) noexcept;
int cuda_validate_blocks(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_blocks_parameters config, cuda_blocks_workspace work, void* stream) noexcept;
}
