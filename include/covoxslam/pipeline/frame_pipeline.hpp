#pragma once

#include <covoxslam/core/result.hpp>
#include <covoxslam/input/dataset.hpp>
#include <covoxslam/pipeline/submap.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <type_traits>
#include <vector>

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

operation_result<std::vector<ray>> deskew_frame(
    const organized_frame& frame, std::span<const pose_sample> trajectory,
    const pose4& anchor, std::size_t ring_stride = 1, std::size_t column_stride = 1);

operation_result<pipeline_result> run_frame_pipeline(
    const dataset& input, const pipeline_options& options = {},
     const pipeline_callbacks& callbacks = {});

static_assert(std::is_aggregate_v<ray>);
static_assert(std::is_standard_layout_v<ray>);
static_assert(std::is_trivially_copyable_v<ray>);
static_assert(sizeof(ray) == 72);
static_assert(alignof(ray) == alignof(std::uint64_t));
static_assert(offsetof(ray, origin) == 0);
static_assert(offsetof(ray, stamp_ns) == 32);
static_assert(offsetof(ray, column) == 48);
static_assert(offsetof(ray, measured_range_m) == 64);
static_assert(std::is_aggregate_v<pipeline_options>);
static_assert(std::is_aggregate_v<pipeline_callbacks>);
static_assert(std::is_aggregate_v<pipeline_result>);
}
