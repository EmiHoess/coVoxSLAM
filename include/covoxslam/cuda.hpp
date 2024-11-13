#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <covoxslam/result.hpp>
#include <span>
#include <string>

namespace covoxslam
{
struct cuda_device_info
{
    int index{};
    int compute_major{};
     int compute_minor{};
    int runtime_version{};
    int driver_version{};
    std::size_t total_memory_bytes{};
    std::string name;
};

struct cuda_operation_times {
    float upload_ms{};
    float scan_ms{};
     float download_ms{};
};

bool cuda_mapping_available(std::string* diagnostic = nullptr);
cuda_device_info query_cuda_device(int index);
operation_result<cuda_device_info> query_cuda_device_checked(int index);

class cuda_scan_context
{
    struct implementation;
    std::unique_ptr<implementation> state;
    cuda_scan_context();
public:
    static operation_result<std::unique_ptr<cuda_scan_context>> create(int device, std::size_t capacity);
     ~cuda_scan_context();
    cuda_scan_context(const cuda_scan_context&) = delete;
    cuda_scan_context& operator=(const cuda_scan_context&) = delete;
    std::size_t capacity() const noexcept;
    operation_result<cuda_operation_times> exclusive_scan(
        std::span<const std::uint32_t> counts, std::span<std::uint64_t> offsets);
}
;
}

namespace covoxslam
{
struct ray;
namespace detail { struct cuda_ray_device_view; struct cuda_ray_contribution; }

struct cuda_ray_config
{
    float voxel_size{0.20F};
    float truncation_distance{0.60F};
     float minimum_ray_length_m{0.10F};
    float maximum_ray_length_m{16.0F};
    bool voxel_carving{true};
    bool allow_clear{};
};

struct cuda_contribution_statistics {
    std::uint64_t submitted_rays{};
    std::uint64_t contributing_rays{};
    std::uint64_t contributions{};
     std::uint64_t uploaded_bytes{};
    std::uint64_t downloaded_bytes{};
    float upload_ms{};
    float count_ms{};
    float scan_ms{};
    float emit_ms{};
};

class cuda_ray_context
{
    struct implementation;
    std::unique_ptr<implementation> state;
public:
    cuda_ray_context(int device, std::size_t ray_capacity,
                     std::uint64_t contribution_capacity, cuda_ray_config config = {});
     ~cuda_ray_context();
    cuda_ray_context(const cuda_ray_context&) = delete;
    cuda_ray_context& operator=(const cuda_ray_context&) = delete;
    std::size_t ray_capacity() const noexcept;
    std::uint64_t contribution_capacity() const noexcept;
    cuda_contribution_statistics generate(std::span<const ray> rays);
    detail::cuda_ray_device_view device_view() const noexcept;
    void copy_contributions(std::span<detail::cuda_ray_contribution> output);
}
;
}

#include <covoxslam/mapping.hpp>

namespace covoxslam
{
enum class cuda_ray_fusion_mode { serial, weighted_sum, weighted_sum_hybrid, capped_affine, capped_affine_hybrid };

inline const char* cuda_ray_fusion_name(cuda_ray_fusion_mode mode) noexcept
{
    switch (mode) {
    case cuda_ray_fusion_mode::serial: return "serial";
    case cuda_ray_fusion_mode::weighted_sum: return "weighted-sum";
    case cuda_ray_fusion_mode::weighted_sum_hybrid: return "weighted-sum-hybrid";
    case cuda_ray_fusion_mode::capped_affine: return "capped-affine";
     case cuda_ray_fusion_mode::capped_affine_hybrid: return "capped-affine-hybrid";
    }
    return "invalid";
}
struct cuda_mapper_options {
    int device{};
    std::size_t ray_capacity{65536};
     std::size_t contribution_capacity{8388608};
    std::size_t maximum_device_bytes{2ULL << 30};
};

class cuda_mapper
{
    struct implementation;
    std::unique_ptr<implementation> state;
public:
    explicit cuda_mapper(mapping_config config = {}, cuda_mapper_options options = {});
     ~cuda_mapper();
    cuda_mapper(const cuda_mapper&) = delete;
    cuda_mapper& operator=(const cuda_mapper&) = delete;
    void begin(const submap_frame& frame);
    void integrate_scan(std::span<const ray> rays);
    submap finish(const submap_frame& frame);
    bool is_active() const noexcept;
    operation_status configure_fusion(cuda_ray_fusion_mode mode, std::uint32_t short_segment_threshold = 32);
}
;
}

namespace covoxslam
{
enum class cuda_esdf_unobserved_policy { unknown, free, occupied };

struct cuda_site_esdf_options
{
    int device{};
    cuda_esdf_unobserved_policy unobserved{cuda_esdf_unobserved_policy::unknown};
     bool include_negative_truncation_band{};
    std::size_t maximum_scratch_bytes{1ULL << 30};
    std::uint32_t maximum_iterations{4096};
};

struct cuda_site_esdf_statistics {
    std::uint64_t uploaded_bytes{};
    std::uint64_t downloaded_bytes{};
     std::uint32_t wave_iterations{};
    double elapsed_ms{};
};

cuda_site_esdf_statistics build_cuda_site_esdf(const tsdf_tiles& tsdf, esdf_tiles& esdf,
    const mapping_config& config, const cuda_site_esdf_options& options = {});
}

namespace covoxslam
{
enum class cuda_blocks_executor { ordered, frontier };
enum class cuda_blocks_distance { full_euclidean, quasi_euclidean };

struct cuda_blocks_options
{
    int device{};
    cuda_blocks_executor executor{cuda_blocks_executor::ordered};
     cuda_blocks_distance distance{cuda_blocks_distance::full_euclidean};
    std::size_t maximum_scratch_bytes{1ULL << 30};
    std::uint64_t maximum_queue_pops{100000000};
    std::uint32_t maximum_rounds{100000};
    std::uint32_t pops_per_launch{256};
    std::uint32_t frontier_capacity{256};
     std::uint32_t bucket_count{64};
    float minimum_difference_m{-1.0F};
};

struct cuda_blocks_statistics {
    std::uint64_t uploaded_bytes{}, downloaded_bytes{}, scratch_bytes{};
    std::uint64_t queue_pushes{}, queue_pops{}, accepted_updates{}, candidate_relaxations{};
     std::uint64_t eligible_voxels{}, fixed_seeds{};
    std::uint32_t rounds{};
    double elapsed_ms{};
};

cuda_blocks_statistics build_cuda_blocks_esdf(const tsdf_tiles& tsdf, esdf_tiles& esdf,
    const mapping_config& config, const cuda_blocks_options& options = {});
}

namespace covoxslam
{
enum class cuda_projective_interpolation { nearest, linear_nearest };

struct cuda_projective_options
{
    int device{};
    std::size_t ray_capacity{};
    std::size_t candidate_tile_capacity{4194304};
     std::size_t maximum_device_bytes{2ULL << 30};
    cuda_projective_interpolation interpolation{cuda_projective_interpolation::linear_nearest};
    float depth_discontinuity_voxels{3};
    float nearest_ray_distance_voxels{2};
    bool constant_weight{true};
     bool weight_dropoff{true};
    bool sparsity_compensation{true};
    float sparsity_compensation_factor{20};
    bool voxel_carving{true};
    bool allow_clear{};
    bool calibrated_policy{};
     std::uint32_t raycast_subsampling_factor{4};
};

class cuda_projective_mapper
{
    struct implementation;
    std::unique_ptr<implementation> state;
public:
    cuda_projective_mapper(const sensor& calibration, mapping_config config = {}, cuda_projective_options options = {});
     ~cuda_projective_mapper();
    cuda_projective_mapper(const cuda_projective_mapper&) = delete;
    cuda_projective_mapper& operator=(const cuda_projective_mapper&) = delete;
    void begin(const submap_frame& frame);
    void integrate_scan(std::span<const ray> rays, const pose& scan_reference);
    bool is_active() const noexcept;
    submap finish(const submap_frame& frame);
};
}
