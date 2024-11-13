#pragma once

#include <cstddef>
#include <cstdint>

namespace covoxslam::detail
{
inline constexpr std::uint32_t cuda_mapping_abi_version = 1;

struct cuda_device_ray {
    float origin[3]{};
    float endpoint[3]{};
     float weight{1.0F};
    std::uint64_t stamp_ns{};
    std::uint32_t frame_id{};
    std::uint32_t column{};
    std::uint32_t ring{};
    std::uint32_t ordinal{};
};

int cuda_scan_counts(const std::uint32_t* counts, std::uint64_t* offsets,
                     std::uint64_t* block_sums, std::size_t count, void* stream) noexcept;
}

namespace covoxslam::detail
{
struct cuda_ray_parameters
{
    float voxel_size{0.20F};
    float truncation_distance{0.60F};
     float minimum_ray_length_m{0.10F};
    float maximum_ray_length_m{16.0F};
    bool voxel_carving{true};
     bool allow_clear{};
};

struct cuda_ray_contribution {
    std::int32_t voxel[3]{};
    float distance{};
    float weight{};
     std::uint32_t ray_ordinal{};
    std::uint32_t traversal_ordinal{};
};

int cuda_count_ray_contributions(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, std::uint32_t* counts, std::uint32_t* status, void* stream) noexcept;
int cuda_emit_ray_contributions(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, const std::uint32_t* counts, const std::uint64_t* offsets,
    cuda_ray_contribution* output, std::uint64_t capacity, std::uint32_t* status, void* stream) noexcept;
}

namespace covoxslam::detail
{
struct cuda_ray_summary
{
    std::uint64_t contributions{};
     std::uint32_t contributing_rays{};
    std::uint32_t status{};
};

struct cuda_ray_device_view {
    cuda_ray_contribution* contributions{};
    std::uint64_t count{};
    std::uint64_t capacity{};
     std::uint32_t* counts{};
    std::uint64_t* offsets{};
    std::uint64_t* block_sums{};
    void* stream{};
    int device{};
    std::size_t ray_capacity{};
};
}

namespace covoxslam::detail
{
int cuda_summarize_ray_contributions(const std::uint32_t* counts, const std::uint64_t* offsets,
    std::size_t count, cuda_ray_summary* summary, void* stream) noexcept;
}

namespace covoxslam::detail
{
struct cuda_tile_coordinate
{
    std::int32_t x{}, y{}, z{};
};

struct cuda_tsdf_voxel {
    float distance{}, weight{};
     std::uint32_t sample_count{}, flags{};
};

struct cuda_tsdf_tile
{
    cuda_tsdf_voxel voxels[512]{};
};

struct cuda_voxel_segment {
    std::int32_t voxel[3]{};
    std::uint32_t count{};
     std::uint64_t begin{};
    std::uint32_t tile_slot{UINT32_MAX};
    std::uint32_t local_index{};
};

struct cuda_fusion_parameters
{
    float truncation_distance{0.60F};
    float maximum_weight{10000.0F};
     float extraction_distance{0.20F};
};

static_assert(sizeof(cuda_tile_coordinate) == 12 && sizeof(cuda_tsdf_voxel) == 16);
static_assert(sizeof(cuda_tsdf_tile) == 8192 && sizeof(cuda_voxel_segment) == 32);

int cuda_stable_sort_contributions(cuda_ray_contribution* values, cuda_ray_contribution* scratch,
    std::uint32_t* flags, std::uint64_t* offsets, std::uint64_t* block_sums,
    std::size_t count, void* stream) noexcept;
int cuda_build_voxel_segments(const cuda_ray_contribution* values, std::size_t count,
    std::uint32_t* flags, std::uint64_t* offsets, std::uint64_t* block_sums,
    cuda_voxel_segment* segments, std::uint32_t capacity, std::uint32_t* segment_count,
    std::uint32_t* status, void* stream) noexcept;
int cuda_fuse_voxel_segments(const cuda_ray_contribution* values, std::size_t count,
    const cuda_voxel_segment* segments, std::uint32_t segment_count,
    const cuda_tile_coordinate* keys, cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    cuda_fusion_parameters config, std::uint32_t* status, void* stream) noexcept;
}

namespace covoxslam::detail
{
enum class cuda_esdf_unobserved : std::uint32_t { unknown, free, occupied };

struct cuda_site_esdf_parameters
{
    float voxel_size{0.20F};
    float truncation_distance{0.60F};
     float site_band_m{0.20F};
    float minimum_weight{1.0F};
    float maximum_distance_m{8.0F};
    cuda_esdf_unobserved unobserved{cuda_esdf_unobserved::unknown};
    bool include_negative_truncation_band{};
};

struct cuda_site_esdf_voxel {
    float squared_distance_voxels{};
    std::int32_t parent[3]{};
     std::uint32_t flags{};
}
;

inline constexpr std::uint32_t cuda_esdf_observed = 1;
inline constexpr std::uint32_t cuda_esdf_inside = 2;
inline constexpr std::uint32_t cuda_esdf_site = 4;
inline constexpr std::uint32_t cuda_esdf_parent_assigned = 8;

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
struct cuda_voxblox_voxel
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

struct cuda_voxblox_queue
{
    std::uint32_t size{}, cursor{}, status{}, frontier_count{};
    std::uint64_t pushes{}, pops{}, accepted{}, relaxations{}, eligible{}, seeds{};
};

struct cuda_voxblox_proposal {
    std::uint32_t target{UINT32_MAX};
    float source_distance_m{}, step_distance_m{};
     std::int32_t parent[3]{};
};

struct cuda_blocks_workspace
{
    cuda_voxblox_voxel* values{};
    std::uint32_t* next{};
    std::uint32_t* queued{};
    std::uint32_t* heads{};
     std::uint32_t* tails{};
    const std::uint32_t* order{};
    std::uint32_t* sources{};
    std::uint32_t* winners{};
    cuda_voxblox_proposal* proposals{};
    cuda_voxblox_queue* queue{};
};

static_assert(sizeof(cuda_voxblox_voxel) == 20 && sizeof(cuda_voxblox_queue) == 64);
static_assert(sizeof(cuda_voxblox_proposal) == 24);
int cuda_initialize_voxblox(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_blocks_parameters config, cuda_blocks_workspace work, void* stream) noexcept;
int cuda_enqueue_voxblox_seeds(std::size_t count, cuda_blocks_parameters config,
    cuda_blocks_workspace work, void* stream) noexcept;
int cuda_propagate_voxblox_ordered(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_blocks_parameters config, cuda_blocks_workspace work, std::uint32_t maximum_pops, void* stream) noexcept;
int cuda_propagate_voxblox_frontier(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_blocks_parameters config, cuda_blocks_workspace work, std::uint32_t capacity, void* stream) noexcept;
int cuda_validate_voxblox(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_blocks_parameters config, cuda_blocks_workspace work, void* stream) noexcept;
}

namespace covoxslam::detail
{
struct cuda_projective_pose
{
    float translation[3]{};
    float rotation_xyzw[4]{0, 0, 0, 1};
};

struct cuda_projective_calibration {
    std::uint32_t rings{};
    std::uint32_t columns{};
     float azimuth_start_rad{};
    float azimuth_coverage_rad{6.283185307179586F};
    bool counter_clockwise{true};
};

bool valid_projective_shape(cuda_projective_calibration calibration) noexcept;
bool valid_projective_pose(cuda_projective_pose pose) noexcept;
bool valid_projective_calibration(cuda_projective_calibration calibration,
    const float* vertical_angles, std::size_t angle_count) noexcept;
}

namespace covoxslam::detail
{
int cuda_build_projective_range_image(const cuda_device_ray* rays, std::size_t count,
    cuda_ray_parameters parameters, cuda_projective_pose pose, cuda_projective_calibration calibration,
    const float* vertical_angles, std::uint64_t* winners, std::size_t pixel_capacity,
    std::uint32_t* status, void* stream) noexcept;
}

namespace covoxslam::detail
{
struct cuda_projective_discovery_options
{
    cuda_projective_calibration calibration;
    const float* angles{};
    bool calibrated_rays{};
     std::uint32_t factor{1};
};

int cuda_count_projective_tiles(const cuda_device_ray* rays, std::size_t ray_count,
    const std::uint64_t* winners, std::size_t pixel_count, cuda_projective_pose pose,
    cuda_ray_parameters parameters, std::uint32_t* counts, std::uint32_t* status, void* stream,
    cuda_projective_discovery_options options = {}) noexcept;
int cuda_emit_projective_tiles(const cuda_device_ray* rays, std::size_t ray_count,
    const std::uint64_t* winners, std::size_t pixel_count, cuda_projective_pose pose,
    cuda_ray_parameters parameters, const std::uint32_t* counts, const std::uint64_t* offsets,
    cuda_tile_coordinate* keys, std::uint64_t capacity, std::uint32_t* status, void* stream,
    cuda_projective_discovery_options options = {}) noexcept;
}

namespace covoxslam::detail
{
enum class cuda_projective_interpolation { nearest, linear_nearest };
struct cuda_projective_parameters
{
    cuda_projective_interpolation interpolation{cuda_projective_interpolation::linear_nearest};
    float depth_discontinuity_voxels{3};
     float nearest_ray_distance_voxels{2};
    bool constant_weight{true};
    bool weight_dropoff{true};
     bool sparsity_compensation{true};
    float sparsity_compensation_factor{20};
    bool projected_policy{};
};
}

namespace covoxslam::detail
{
int cuda_fuse_projective_tiles(const std::uint32_t* slots, std::size_t slot_count,
    const cuda_tile_coordinate* keys, cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    const cuda_device_ray* rays, std::size_t ray_count, const std::uint64_t* winners, const float* angles,
    cuda_projective_pose pose, cuda_projective_calibration calibration, cuda_ray_parameters ray_parameters,
    cuda_fusion_parameters fusion, cuda_projective_parameters sampling,
    std::uint64_t* updates, std::uint32_t* status, void* stream) noexcept;
}

namespace covoxslam::detail
{
enum class cuda_ray_fusion_mode { serial, weighted_sum, weighted_sum_hybrid, capped_affine, capped_affine_hybrid };

struct cuda_ray_fusion_options
{
    cuda_ray_fusion_mode mode{cuda_ray_fusion_mode::serial};
    std::uint32_t short_segment_threshold{32};
};
}

namespace covoxslam::detail
{
int cuda_reduce_voxel_segments(const cuda_ray_contribution* values, std::size_t count,
    const cuda_voxel_segment* segments, std::uint32_t segment_count,
    const cuda_tile_coordinate* keys, cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    cuda_fusion_parameters config, cuda_ray_fusion_options options,
    std::uint32_t* status, void* stream) noexcept;
}
