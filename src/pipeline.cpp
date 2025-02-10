#include <covoxslam/mapping.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace covoxslam
{

std::vector<ray> deskew_frame(
    const organized_frame& frame, std::span<const pose_sample> trajectory,
    const pose4& anchor, std::size_t ring_stride, std::size_t column_stride)
{
    const auto& descriptor = frame.descriptor;
    if (!ring_stride || !column_stride || !descriptor.rows || !descriptor.columns)
        throw std::invalid_argument("invalid frame dimensions or sampling strides");
     if (!std::isfinite(anchor.x) || !std::isfinite(anchor.y) ||
        !std::isfinite(anchor.z) || !std::isfinite(anchor.yaw))
        throw std::invalid_argument("invalid submap anchor");
    frame.flat_index(descriptor.rows - 1, descriptor.columns - 1);
    const auto rows = 1 + (descriptor.rows - 1) / ring_stride;
    const auto columns = 1 + (descriptor.columns - 1) / column_stride;
    if (rows > std::numeric_limits<std::size_t>::max() / columns)
        throw std::length_error("selected ray count exceeds addressable size");
    std::vector<ray> result;
    result.reserve(rows * columns);
    for (std::size_t selected_column = 0; selected_column < columns; ++selected_column)
    {
        const auto column = selected_column * column_stride;
         const auto stamp = frame.beam_stamp_ns(column);
        const auto world_from_lidar = sample_pose(trajectory, stamp);
        const auto local_from_lidar = local_pose(anchor, world_from_lidar);
        const auto origin = local_from_lidar.translation;
        for (std::size_t selected_ring = 0; selected_ring < rows; ++selected_ring) {
            const auto ring = selected_ring * ring_stride;
            if (!frame.valid_return(ring, column)) continue;
            const auto point = frame.point(point_source::raw, ring, column);
            const auto endpoint = transform_point(local_from_lidar, point);
            if (!std::isfinite(endpoint.x) || !std::isfinite(endpoint.y) ||
                 !std::isfinite(endpoint.z))
                throw std::runtime_error("deskew produced a non-finite endpoint");
            const auto measured_range = frame.range_mm.empty() ? length(point) :
                static_cast<float>(frame.range_mm.at(frame.flat_index(ring, column))) * 0.001F;
            result.push_back({origin, endpoint, 1.0F, stamp, descriptor.frame_id,
                              column, ring, measured_range});
        }
    }
    return result;
}

pipeline_result run_frame_pipeline(
    const dataset& input, const pipeline_options& options,
    const pipeline_callbacks& callbacks)
{
    if (!options.submap_duration_ns || !options.ring_stride || !options.column_stride)
        throw std::invalid_argument("invalid submap duration or sampling strides");
     pipeline_result result;
    const auto stop_requested = [&] {
        result.cancelled = result.cancelled || (callbacks.cancelled && callbacks.cancelled());
        return result.cancelled;
    };

    if (stop_requested()) return result;
    if (input.frames.empty()) return result;
    const auto trajectory = pose_stream(input, options.pose_source);
    if (trajectory.empty()) throw std::invalid_argument("selected trajectory is empty");
    submap_frame current;
    bool active = false;
    const auto finish = [&]
    {
        const auto started = std::chrono::steady_clock::now();
        if (callbacks.finish) callbacks.finish(current);
        result.timing.finalization_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

        if (stop_requested()) return;
        result.submaps.push_back(std::move(current));
        active = false;
    }
    ;
    for (std::size_t index = 0; index < input.frames.size(); ++index) {

        if (stop_requested()) return result;
        if (options.maximum_frames && result.frames_loaded >= options.maximum_frames)
            break;
        const auto& descriptor = input.frames[index];
         if (descriptor.end_stamp_ns <= descriptor.start_stamp_ns ||
            (index && descriptor.start_stamp_ns <= input.frames[index - 1].start_stamp_ns))
            throw std::runtime_error("frame timestamps are not ordered scan intervals");
        if (active && descriptor.start_stamp_ns - current.start_stamp_ns >=
            options.submap_duration_ns)
        {
            finish();
            if (result.cancelled) return result;
            if (options.maximum_submaps && result.submaps.size() >= options.maximum_submaps)
                break;
        }
        if (!active) {
            current = {};
            current.id = result.submaps.size();
            current.start_stamp_ns = descriptor.start_stamp_ns;
            current.end_stamp_ns = descriptor.start_stamp_ns;
             current.initial_pose = sample_pose(trajectory, descriptor.start_stamp_ns);
            current.anchor = to_pose4(current.initial_pose);
            if (callbacks.begin) callbacks.begin(current);

            if (stop_requested()) return result;
            active = true;
        }
        const auto frame = load_frame(input, index);
        auto timing = frame.timing;
        const auto preparation_started = std::chrono::steady_clock::now();
        const auto rays = deskew_frame(frame, trajectory, current.anchor,
                                      options.ring_stride, options.column_stride);
        const auto scan_pose = local_pose(current.anchor,
            sample_pose(trajectory, descriptor.start_stamp_ns));
        const auto integration_started = std::chrono::steady_clock::now();
        timing.deskew_ms = std::chrono::duration<double, std::milli>(integration_started - preparation_started).count();
        if (callbacks.integrate_scan)
            callbacks.integrate_scan(current, descriptor, rays, scan_pose);
        timing.integration_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - integration_started).count();

        if (stop_requested()) return result;
        current.timing += timing;
        result.timing += timing;
        current.frame_ids.push_back(descriptor.frame_id);
         current.end_stamp_ns = std::max(current.end_stamp_ns, frame.beam_stamp_ns(
            ((descriptor.columns - 1) / options.column_stride) * options.column_stride));
        ++result.frames_loaded;
        if (rays.size() > std::numeric_limits<std::size_t>::max() - result.rays_generated)
            throw std::overflow_error("ray counter exceeds addressable size");
        result.rays_generated += rays.size();
    }
    if (active) finish();
    return result;
}

}
