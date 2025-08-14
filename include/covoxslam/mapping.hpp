#pragma once

#include <covoxslam/core/geometry.hpp>
#include <covoxslam/input/dataset.hpp>
#include <covoxslam/input_timing.hpp>
#include <covoxslam/voxel_records.h>

#include <functional>
#include <span>

namespace covoxslam
{

struct ray
{
    vec3f origin{};
    vec3f endpoint{};
    float weight{1.0F};
     std::uint64_t stamp_ns{};
    int frame_id{};
    std::size_t column{};
    std::size_t ring{};
    float measured_range_m{};
};

struct pipeline_options {
    covoxslam::pose_source pose_source{covoxslam::pose_source::odometry};
    std::uint64_t submap_duration_ns{10'000'000'000ULL};
    std::size_t ring_stride{1};
     std::size_t column_stride{1};
    std::size_t maximum_frames{};
    std::size_t maximum_submaps{};
};

struct submap_frame
{
    std::size_t id{};
    std::uint64_t start_stamp_ns{};
    std::uint64_t end_stamp_ns{};
     pose4 anchor{};
    pose initial_pose{};
    std::vector<int> frame_ids;
    input_timing timing;
};

struct pipeline_callbacks
{
    std::function<void(const submap_frame&)> begin;
    std::function<void(const submap_frame&, const frame_descriptor&,
                       std::span<const ray>, const pose&)> integrate_scan;
    std::function<void(const submap_frame&)> finish;
    std::function<bool()> cancelled;
};

struct pipeline_result {
    std::size_t frames_loaded{};
     std::size_t rays_generated{};
    std::vector<submap_frame> submaps;
    bool cancelled{};
    input_timing timing;
}
;

std::vector<ray> deskew_frame(
    const organized_frame& frame, std::span<const pose_sample> trajectory,
    const pose4& anchor, std::size_t ring_stride = 1, std::size_t column_stride = 1);

operation_result<pipeline_result> run_frame_pipeline(
    const dataset& input, const pipeline_options& options = {},
     const pipeline_callbacks& callbacks = {});

}

#include <array>
#include <unordered_map>

namespace covoxslam
{
inline constexpr int tile_edge = 8;
inline constexpr std::size_t tile_voxels = 512;
inline constexpr std::uint32_t observed_flag = 1;
inline constexpr std::uint32_t near_surface_flag = 2;
inline constexpr std::uint32_t esdf_site_flag = 4;

using tsdf_voxel = tsdf_record;
using esdf_voxel = esdf_record;

struct tsdf_tile
{
    std::array<tsdf_voxel, tile_voxels> voxels{};
};

struct esdf_tile
{
     std::array<esdf_voxel, tile_voxels> voxels{};
};

struct int3_hash {
    std::size_t operator()(int3 value) const noexcept;
};

using tsdf_tiles = std::unordered_map<int3, tsdf_tile, int3_hash>;
using esdf_tiles = std::unordered_map<int3, esdf_tile, int3_hash>;

int3 block_index(int3 voxel) noexcept;
int3 global_index(int3 block, std::size_t local);
vec3f voxel_center(int3 index, float size) noexcept;
std::size_t local_index(int3 voxel) noexcept;
bool block_index_less(int3 left, int3 right) noexcept;
const tsdf_voxel* find_voxel(const tsdf_tiles& tiles, int3 voxel) noexcept;
}

#include <optional>

namespace covoxslam
{
enum class esdf_backend { none = 0, cpu_blocks_batch = 1, cuda_projective_block_band = 2, cuda_blocks_batch = 3, cuda_block_band_soa = 4 };
const char* esdf_backend_name(esdf_backend value) noexcept;

struct mapping_config
{
    float voxel_size{0.20F};
    float truncation_distance{0.60F};
    float extraction_distance{0.20F};
     float maximum_weight{10000.0F};
    float minimum_ray_length_m{0.10F};
    float maximum_ray_length_m{16.0F};
    std::size_t maximum_tiles{32768};
    float esdf_site_band_m{0.20F};
    float esdf_maximum_distance_m{8.0F};
     float esdf_minimum_weight{1.0F};
    esdf_backend esdf_generator{esdf_backend::cuda_block_band_soa};
    bool valid() const noexcept;
};

struct mapping_statistics {
    std::uint64_t rays_submitted{};
    std::uint64_t rays_integrated{};
    std::uint64_t rays_rejected{};
    std::uint64_t voxel_updates{};
     std::uint64_t allocated_tiles{};
    double integration_ms{};
    double esdf_ms{};
    std::uint64_t cuda_uploaded_bytes{};
    std::uint64_t cuda_downloaded_bytes{};
    double sensor_gather_ms{}, sensor_mask_ms{}, sensor_discovery_ms{}, sensor_fusion_ms{};
    double sensor_candidates_ms{};
    double sensor_gpu_discovery_ms{}, sensor_gpu_allocation_ms{};
    double sensor_resident_dispatch_ms{};
    std::uint64_t sensor_candidate_overflows{};
    double sensor_graph_setup_ms{}, sensor_upload_ms{}, sensor_readback_ms{};
    std::uint64_t sensor_candidate_voxels{}, sensor_kernel_launches{};
    std::uint32_t sensor_transfer_bits{};
    std::uint32_t cuda_accumulator_bits{};
}
;

struct submap
{
    submap_frame frame;
    pose4 optimized_world_pose{};
    tsdf_tiles tsdf;
     esdf_tiles esdf;
    mapping_config config;
    mapping_statistics statistics;
};

}
