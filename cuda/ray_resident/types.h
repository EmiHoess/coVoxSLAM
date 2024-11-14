#pragma once

#include "../sparse_mapping.h"

namespace covoxslam::detail
{
enum cuda_resident_error : std::uint32_t {
    resident_coordinate_error = 1U,
    resident_hash_capacity_error = 2U,
    resident_tile_capacity_error = 4U,
     resident_contribution_capacity_error = 8U,
    resident_scatter_error = 16U,
    resident_traversal_error = 32U,
    resident_numeric_error = 64U
};

struct cuda_resident_hash_entry
{
    std::uint64_t key{UINT64_MAX};
    std::uint32_t slot{UINT32_MAX}, reserved{};
};

struct cuda_resident_contribution {
    std::uint64_t order{};
    float distance{}, weight{};
};

struct cuda_resident_state
{
    std::uint32_t error{}, tile_count{}, maximum_segment_length{};
     std::uint32_t batch_integrated{}, batch_rejected{};
    std::uint64_t batch_contributions{}, batch_touched{}, batches{};
    std::uint64_t integrated{}, rejected{}, contributions{}, touched{};
};

struct cuda_resident_view
{
    cuda_resident_state* state{};
    cuda_resident_hash_entry* hash{};
     std::uint32_t hash_capacity{}, tile_capacity{};
    cuda_tile_coordinate* keys{};
    cuda_tsdf_tile* tiles{};
    std::uint32_t* counts{};
    std::uint32_t* cursors{};
    std::uint64_t* offsets{};
    std::uint64_t* block_sums{};
    cuda_resident_contribution* contributions{};
     std::uint64_t contribution_capacity{};
};

static_assert(sizeof(cuda_resident_hash_entry) == 16);
static_assert(sizeof(cuda_resident_contribution) == 16);
int cuda_initialize_resident_map(cuda_resident_view view, void* stream) noexcept;
}
