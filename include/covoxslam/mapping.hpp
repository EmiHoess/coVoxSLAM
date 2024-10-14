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
