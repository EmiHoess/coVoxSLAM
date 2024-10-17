#pragma once

#include <covoxslam/dataset.hpp>

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
};

struct pipeline_callbacks
{
    std::function<void(const submap_frame&)> begin;
    std::function<void(const submap_frame&, const frame_descriptor&,
                       std::span<const ray>, const pose&)> integrate_scan;
    std::function<void(const submap_frame&)> finish;
};

struct pipeline_result {
    std::size_t frames_loaded{};
     std::size_t rays_generated{};
    std::vector<submap_frame> submaps;
}
;

std::vector<ray> deskew_frame(
    const organized_frame& frame, std::span<const pose_sample> trajectory,
    const pose4& anchor, std::size_t ring_stride = 1, std::size_t column_stride = 1);

pipeline_result run_frame_pipeline(
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

struct tsdf_voxel
{
    float distance{};
    float weight{};
     std::uint32_t sample_count{};
    std::uint32_t flags{};
};

struct esdf_voxel {
    float signed_distance_m{};
    std::int32_t parent_dx{};
    std::int32_t parent_dy{};
    std::int32_t parent_dz{};
    std::uint32_t flags{};
}
;

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
int3 voxel_index(vec3f point, float voxel_size);
int3 global_index(int3 block, std::size_t local);
vec3f voxel_center(int3 index, float size) noexcept;
std::size_t local_index(int3 voxel) noexcept;
bool block_index_less(int3 left, int3 right) noexcept;
const tsdf_voxel* find_voxel(const tsdf_tiles& tiles, int3 voxel) noexcept;
tsdf_voxel& allocate_voxel(tsdf_tiles& tiles, int3 voxel, std::size_t maximum_tiles);
}

#include <optional>

namespace covoxslam
{
struct submap
{
    submap_frame frame;
    pose4 optimized_world_pose{};
    tsdf_tiles tsdf;
     esdf_tiles esdf;
};

class cpu_mapper
{
    std::optional<submap> active;
public:
    void begin(const submap_frame& frame);
    submap finish(const submap_frame& frame);
    bool is_active() const noexcept;
}
;
}
