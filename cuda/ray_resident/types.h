#pragma once

#include "../sparse_mapping.h"
#include "../common/tile_key.h"

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

struct cuda_resident_contribution {
    std::uint64_t order{};
    float distance{}, weight{};
    std::uint32_t input_index{};
};

struct cuda_resident_state
{
    std::uint32_t error{}, tile_count{}, maximum_segment_length{};
     std::uint32_t batch_integrated{}, batch_rejected{};
    std::uint64_t batch_contributions{}, batch_touched{}, batches{};
    std::uint64_t integrated{}, rejected{}, contributions{}, touched{};
    std::uint64_t batch_validated{};
    std::uint32_t batch_open{}, batch_reduction{};
    std::uint32_t batch_task_phase{}, batch_compact_rays{};
    unsigned long long batch_warp_counts[4]{}, warp_counts[4]{};
    unsigned long long batch_skipped{}, skipped{};
};

struct cuda_resident_view
{
    cuda_resident_state* state{};
    cuda_resident_hash_entry* hash{};
     std::uint32_t hash_capacity{}, tile_capacity{};
    cuda_tile_coordinate* keys{};
    cuda_tsdf_tile* tiles{};
    cuda_tsdf_voxel* pending{};
    std::uint32_t* counts{};
    std::uint32_t* cursors{};
    std::uint64_t* offsets{};
    std::uint64_t* block_sums{};
    cuda_resident_contribution* contributions{};
     std::uint64_t contribution_capacity{};
};

static_assert(sizeof(cuda_resident_contribution) == 24);
inline bool valid_resident_view(cuda_resident_view view) noexcept
{
    return view.state && view.hash && view.keys && view.tiles && view.pending && view.counts && view.cursors &&
        view.offsets && view.block_sums && view.contributions && view.contribution_capacity &&
        view.tile_capacity && view.tile_capacity <= UINT32_MAX / 512 && view.hash_capacity &&
        !(view.hash_capacity & (view.hash_capacity - 1));
}

inline bool valid_resident_map(cuda_resident_view view) noexcept {
    return view.state && view.hash && view.keys && view.tiles && view.pending && view.counts && view.cursors &&
        view.tile_capacity && view.tile_capacity <= UINT32_MAX / 512 && view.hash_capacity &&
        !(view.hash_capacity & (view.hash_capacity - 1));
}

int cuda_initialize_resident_map(cuda_resident_view view, void* stream) noexcept;
int cuda_restore_resident_keys(cuda_resident_view view, std::uint32_t count, void* stream) noexcept;
int cuda_begin_resident_batch(cuda_resident_view view, void* stream) noexcept;
int cuda_finish_resident_batch(cuda_resident_view view, cuda_fusion_parameters parameters, void* stream) noexcept;
int cuda_finish_resident_arrival_batch(cuda_resident_view view, cuda_fusion_parameters parameters, void* stream) noexcept;
int cuda_walk_resident_rays(cuda_resident_view view, const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, unsigned stage, void* stream) noexcept;
}
