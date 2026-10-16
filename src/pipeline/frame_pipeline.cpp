#include <covoxslam/pipeline/frame_pipeline.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

namespace covoxslam
{
namespace
{
inline constexpr float millimeters_to_meters = 0.001F;

operation_status validate_pipeline_options(const pipeline_options& options)
{
    if (!options.submap_duration_ns || !options.ring_stride || !options.column_stride)
        return std::unexpected(operation_error{"invalid submap duration or sampling strides"});
    return {};
}

operation_status validate_pipeline_frame(const dataset& input, std::size_t index)
{
    const auto& descriptor = input.frames[index];
    if (descriptor.end_stamp_ns <= descriptor.start_stamp_ns ||
        (index && descriptor.start_stamp_ns <= input.frames[index - 1].start_stamp_ns))
        return std::unexpected(operation_error{"frame timestamps are not ordered scan intervals"});
    return {};
}

operation_status validate_ray_counter(std::size_t current, std::size_t additional)
{
    if (additional > std::numeric_limits<std::size_t>::max() - current)
        return std::unexpected(operation_error{"ray counter exceeds addressable size"});
    return {};
}

operation_status invoke_begin(const pipeline_callbacks& callbacks, const submap_frame& frame)
{
    if (!callbacks.begin) return {};
    return callbacks.begin(frame);
}

operation_status invoke_integration(const pipeline_callbacks& callbacks,
    const submap_frame& frame, const frame_descriptor& descriptor,
    std::span<const ray> rays, const pose& transform)
{
    if (!callbacks.integrate_scan) return {};
    return callbacks.integrate_scan(frame, descriptor, rays, transform);
}

operation_status invoke_finish(const pipeline_callbacks& callbacks, const submap_frame& frame)
{
    if (!callbacks.finish) return {};
    return callbacks.finish(frame);
}
}

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
                static_cast<float>(frame.range_mm[index]) * millimeters_to_meters;
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
    const auto valid_options = validate_pipeline_options(options);
    if (!valid_options) return std::unexpected(valid_options.error());
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
    const auto finish = [&]() -> operation_status
    {
        const auto started = std::chrono::steady_clock::now();
        const auto finalized = invoke_finish(callbacks, current);
        if (!finalized) return std::unexpected(finalized.error());
        result.timing.finalization_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();

        if (stop_requested()) return {};
        result.submaps.push_back(std::move(current));
        active = false;
        return {};
    }
    ;
    for (std::size_t index = 0; index < input.frames.size(); ++index) {

        if (stop_requested()) return result;
        if (options.maximum_frames && result.frames_loaded >= options.maximum_frames)
            break;
        const auto& descriptor = input.frames[index];
        const auto valid_frame = validate_pipeline_frame(input, index);
        if (!valid_frame) return std::unexpected(valid_frame.error());
        if (active && descriptor.start_stamp_ns - current.start_stamp_ns >=
            options.submap_duration_ns)
        {
            const auto finalized = finish();
            if (!finalized) return std::unexpected(finalized.error());
            if (result.cancelled) return result;
            if (options.maximum_submaps && result.submaps.size() >= options.maximum_submaps)
                break;
        }
        const auto opening_submap = !active;
        if (opening_submap) {
            current = {};
            current.id = result.submaps.size();
            current.start_stamp_ns = descriptor.start_stamp_ns;
            current.end_stamp_ns = descriptor.start_stamp_ns;
            const auto initial_pose = sample_pose(*trajectory, descriptor.start_stamp_ns);
            if (!initial_pose) return std::unexpected(initial_pose.error());
            current.initial_pose = *initial_pose;
            current.anchor = to_pose4(current.initial_pose);
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
        const auto end_stamp = frame->beam_stamp_ns(
            ((descriptor.columns - 1) / options.column_stride) * options.column_stride);
        if (!end_stamp) return std::unexpected(end_stamp.error());
        const auto available_counter = validate_ray_counter(result.rays_generated, rays->size());
        if (!available_counter) return std::unexpected(available_counter.error());
        const auto integration_started = std::chrono::steady_clock::now();
        timing.deskew_ms = std::chrono::duration<double, std::milli>(integration_started - preparation_started).count();
        if (opening_submap) {
            const auto begun = invoke_begin(callbacks, current);
            if (!begun) return std::unexpected(begun.error());

            if (stop_requested()) return result;
            active = true;
        }
        const auto integrated = invoke_integration(callbacks, current, descriptor, *rays, scan_pose);
        if (!integrated) return std::unexpected(integrated.error());
        timing.integration_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - integration_started).count();

        if (stop_requested()) return result;
        current.timing += timing;
        result.timing += timing;
        current.frame_ids.push_back(descriptor.frame_id);
         current.end_stamp_ns = std::max(current.end_stamp_ns, *end_stamp);
        ++result.frames_loaded;
        result.rays_generated += rays->size();
    }
    if (active) {
        const auto finalized = finish();
        if (!finalized) return std::unexpected(finalized.error());
    }
    return result;
}

}
