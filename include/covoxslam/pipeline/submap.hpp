#pragma once

#include <covoxslam/core/geometry.hpp>
#include <covoxslam/input/lidar.hpp>
#include <covoxslam/mapping/config.hpp>
#include <covoxslam/voxel/sparse_map.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace covoxslam
{
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

static_assert(std::is_aggregate_v<submap_frame>);
static_assert(std::is_aggregate_v<mapping_statistics>);
static_assert(std::is_standard_layout_v<mapping_statistics>);
static_assert(std::is_trivially_copyable_v<mapping_statistics>);
static_assert(sizeof(mapping_statistics) == 192);
static_assert(alignof(mapping_statistics) == alignof(double));
static_assert(offsetof(mapping_statistics, rays_submitted) == 0);
static_assert(offsetof(mapping_statistics, allocated_tiles) == 32);
static_assert(offsetof(mapping_statistics, integration_ms) == 40);
static_assert(offsetof(mapping_statistics, sensor_transfer_bits) == 184);
static_assert(offsetof(mapping_statistics, cuda_accumulator_bits) == 188);
static_assert(std::is_aggregate_v<submap>);
}
