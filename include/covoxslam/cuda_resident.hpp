#pragma once

#include <covoxslam/cuda.hpp>
#include <covoxslam/resident_policy.hpp>

namespace covoxslam
{
struct cuda_free_space_options
{
    bool enabled{};
    float minimum_weight{16.f};
     std::uint32_t refresh_scans{8};
};

struct cuda_resident_options
{
    int device{};
    std::uint32_t tile_capacity{4096};
     std::uint32_t ray_capacity{65536};
    std::uint32_t contribution_capacity{8388608};
    std::size_t maximum_device_bytes{2ULL << 30};
    bool voxel_carving{true};
    bool allow_clear{};
    cuda_resident_order order{cuda_resident_order::ordered};
    cuda_resident_reduction reduction{cuda_resident_reduction::none};
    bool step_tasks{};
    cuda_resident_warp_options warp;
    cuda_free_space_options free_space;
    float segment_length{};
};

struct cuda_resident_statistics
{
    mapping_statistics mapping;
    std::uint64_t batches{}, explicit_waits{};
    std::uint64_t warp_flush_groups{}, warp_peer_groups{}, warp_cache_hits{}, warp_cache_flushes{};
    std::uint64_t skipped_contributions{};
     std::uint32_t maximum_segment_length{};
};

class cuda_resident_context
{
    struct implementation;
    std::unique_ptr<implementation> state;
    cuda_resident_context();
public:
    static operation_result<std::unique_ptr<cuda_resident_context>> create(
        mapping_config config = {}, cuda_resident_options options = {});
    ~cuda_resident_context();
    cuda_resident_context(const cuda_resident_context&) = delete;
     cuda_resident_context& operator=(const cuda_resident_context&) = delete;
    std::size_t device_bytes() const noexcept;
    cuda_resident_order fusion_order() const noexcept;
    cuda_resident_reduction reduction_mode() const noexcept;
    bool uses_step_tasks() const noexcept;
    cuda_resident_warp_options warp_configuration() const noexcept;
    cuda_free_space_options free_space_configuration() const noexcept;
    float segment_length() const noexcept;
    operation_status submit(std::span<const ray> rays);
    operation_result<cuda_resident_statistics> flush();
    operation_result<tsdf_tiles> download();
    operation_status reset();
    operation_status restore(const tsdf_tiles& tiles);
};
}
