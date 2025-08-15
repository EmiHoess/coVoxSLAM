#include <covoxslam/mapping.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace covoxslam
{

operation_result<std::vector<ray>> deskew_frame(
    const organized_frame& frame, std::span<const pose_sample> trajectory,
    const pose4& anchor, std::size_t ring_stride, std::size_t column_stride)
{
    const auto& descriptor = frame.descriptor;
    if (!ring_stride || !column_stride || !descriptor.rows || !descriptor.columns)
        return std::unexpected(operation_error{"invalid frame dimensions or sampling strides"});
     if (!std::isfinite(anchor.x) || !std::isfinite(anchor.y) ||
        !std::isfinite(anchor.z) || !std::isfinite(anchor.yaw))
        return std::unexpected(operation_error{"invalid submap anchor"});
    const auto last = frame.flat_index(descriptor.rows - 1, descriptor.columns - 1);
    if (!last) return std::unexpected(last.error());
    const auto rows = 1 + (descriptor.rows - 1) / ring_stride;
    const auto columns = 1 + (descriptor.columns - 1) / column_stride;
    if (rows > std::numeric_limits<std::size_t>::max() / columns)
        return std::unexpected(operation_error{"selected ray count exceeds addressable size"});
    const auto count = *last + 1;
    if (frame.raw_xyz.size() < count)
        return std::unexpected(operation_error{"requested point source is unavailable"});
    if (frame.status.size() < count)
        return std::unexpected(operation_error{"frame status buffer is unavailable"});
    if (!frame.range_mm.empty() && frame.range_mm.size() < count)
        return std::unexpected(operation_error{"frame range buffer is unavailable"});
    if (frame.time_offset_ns.size() < descriptor.columns)
        return std::unexpected(operation_error{"frame time offset buffer is unavailable"});
    for (std::size_t column = 0; column < descriptor.columns; ++column) {
        const auto stamp = frame.beam_stamp_ns(column);
        if (!stamp) return std::unexpected(stamp.error());
    }
    for (std::size_t index = 0; index < count; ++index)
        if (frame.status[index] > static_cast<std::uint8_t>(beam_status::valid_outlier))
            return std::unexpected(operation_error{"frame buffer contains an unknown status code"});
    std::vector<ray> result;
    result.reserve(rows * columns);
    for (std::size_t selected_column = 0; selected_column < columns; ++selected_column)
    {
        const auto column = selected_column * column_stride;
         const auto stamp = descriptor.start_stamp_ns + frame.time_offset_ns[column];
        const auto world_from_lidar = sample_pose(trajectory, stamp);
        if (!world_from_lidar) return std::unexpected(world_from_lidar.error());
        const auto local_from_lidar = local_pose(anchor, *world_from_lidar);
        const auto origin = local_from_lidar.translation;
        for (std::size_t selected_ring = 0; selected_ring < rows; ++selected_ring) {
            const auto ring = selected_ring * ring_stride;
            const auto index = ring * descriptor.columns + column;
            const auto status = frame.status[index];
            if (status != static_cast<std::uint8_t>(beam_status::valid_hit) &&
                status != static_cast<std::uint8_t>(beam_status::valid_outlier)) continue;
            const auto& point = frame.raw_xyz[index];
            const auto endpoint = transform_point(local_from_lidar, point);
            if (!std::isfinite(endpoint.x) || !std::isfinite(endpoint.y) ||
                 !std::isfinite(endpoint.z))
                return std::unexpected(operation_error{"deskew produced a non-finite endpoint"});
            const auto measured_range = frame.range_mm.empty() ? length(point) :
                static_cast<float>(frame.range_mm[index]) * 0.001F;
            result.push_back({origin, endpoint, 1.0F, stamp, descriptor.frame_id,
                              column, ring, measured_range});
        }
    }
    return result;
}

operation_result<pipeline_result> run_frame_pipeline(
    const dataset& input, const pipeline_options& options,
    const pipeline_callbacks& callbacks)
{
    if (!options.submap_duration_ns || !options.ring_stride || !options.column_stride)
        return std::unexpected(operation_error{"invalid submap duration or sampling strides"});
     pipeline_result result;
    const auto stop_requested = [&] {
        result.cancelled = result.cancelled || (callbacks.cancelled && callbacks.cancelled());
        return result.cancelled;
    };

    if (stop_requested()) return result;
    if (input.frames.empty()) return result;
    const auto trajectory = pose_stream(input, options.pose_source);
    if (!trajectory) return std::unexpected(trajectory.error());
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
            return std::unexpected(operation_error{"frame timestamps are not ordered scan intervals"});
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
            const auto initial_pose = sample_pose(*trajectory, descriptor.start_stamp_ns);
            if (!initial_pose) return std::unexpected(initial_pose.error());
            current.initial_pose = *initial_pose;
            current.anchor = to_pose4(current.initial_pose);
            if (callbacks.begin) callbacks.begin(current);

            if (stop_requested()) return result;
            active = true;
        }
        auto frame = load_frame(input, index);
        if (!frame) return std::unexpected(frame.error());
        auto timing = frame->timing;
        const auto preparation_started = std::chrono::steady_clock::now();
        auto rays = deskew_frame(*frame, *trajectory, current.anchor,
                                 options.ring_stride, options.column_stride);
        if (!rays) return std::unexpected(rays.error());
        const auto world_scan_pose = sample_pose(*trajectory, descriptor.start_stamp_ns);
        if (!world_scan_pose) return std::unexpected(world_scan_pose.error());
        const auto scan_pose = local_pose(current.anchor, *world_scan_pose);
        const auto integration_started = std::chrono::steady_clock::now();
        timing.deskew_ms = std::chrono::duration<double, std::milli>(integration_started - preparation_started).count();
        if (callbacks.integrate_scan)
            callbacks.integrate_scan(current, descriptor, *rays, scan_pose);
        timing.integration_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - integration_started).count();

        if (stop_requested()) return result;
        current.timing += timing;
        result.timing += timing;
        current.frame_ids.push_back(descriptor.frame_id);
        const auto end_stamp = frame->beam_stamp_ns(
            ((descriptor.columns - 1) / options.column_stride) * options.column_stride);
        if (!end_stamp) return std::unexpected(end_stamp.error());
         current.end_stamp_ns = std::max(current.end_stamp_ns, *end_stamp);
        ++result.frames_loaded;
        if (rays->size() > std::numeric_limits<std::size_t>::max() - result.rays_generated)
            return std::unexpected(operation_error{"ray counter exceeds addressable size"});
        result.rays_generated += rays->size();
    }
    if (active) finish();
    return result;
}

}
