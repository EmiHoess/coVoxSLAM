#include "active_submap_internal.hpp"
#include <chrono>

namespace covoxslam::sensor_grid
{
operation_result<transfer_times> gpu_map::implementation::integrate_resident(field_view source,
    detail::cuda_projective_pose transform, interpolation mode)
{
    const auto start = std::chrono::steady_clock::now();
    transfer_times result;
    discovery_state counters{};
    std::vector<detail::cuda_tile_coordinate> staging(capacity);
    auto status = check(cudaEventRecord(events[6], stream));
    if (status) status = check(cuda_sensor_discover(source, mask, transform, discovery, stream));
    if (status) status = check(cudaEventRecord(events[7], stream));
    if (status) status = check(cudaEventRecord(events[4], stream));
    if (status) status = check(cuda_sensor_allocate_tiles(resident, stream));
    if (status) status = check(cudaEventRecord(events[5], stream));
    if (status) status = check(cudaMemsetAsync(updates, 0, sizeof(*updates), stream));
    if (status) status = check(cudaEventRecord(events[2], stream));
    if (status) {
        auto descriptor = soa->descriptor();
        if (!descriptor) status = std::unexpected(descriptor.error());
        else status = check(cuda_sensor_resident_soa(source, resident, *descriptor, transform,
            settings.voxel_size, settings.maximum_weight, settings.voxel_size, updates, stream));
    }
    if (status) status = check(cudaEventRecord(events[3], stream));
    if (status) status = check(cudaMemcpyAsync(
        &result.updates, updates, sizeof(result.updates), cudaMemcpyDeviceToHost, stream));
    if (status) status = check(cudaMemcpyAsync(&counters, discovery.state, sizeof(counters), cudaMemcpyDeviceToHost, stream));
    if (status) status = check(cudaMemcpyAsync(staging.data(), keys, staging.size() * sizeof(*keys), cudaMemcpyDeviceToHost, stream));
    status = wait(status);
    if (status) status = check(cudaEventElapsedTime(&result.mask_ms, events[0], events[1]));
    if (status) status = check(cudaEventElapsedTime(&result.gpu_discovery_ms, events[6], events[7]));
    if (status) status = check(cudaEventElapsedTime(&result.gpu_allocation_ms, events[4], events[5]));
    if (status) status = check(cudaEventElapsedTime(&result.fusion_ms, events[2], events[3]));

    if (!status) return std::unexpected(status.error());
    if (counters.error || counters.tile_count > capacity || counters.tile_count < indices.size() || counters.count > capacity) {
        failed = true;
        return std::unexpected(operation_error{"sensor GPU tile allocation failed before publication"});
    }

    for (std::size_t slot = indices.size(); slot < counters.tile_count; ++slot) {
        const auto key = staging[slot];
        const int3 coordinate{key.x, key.y, key.z};

        if (!index.emplace(coordinate, static_cast<std::uint32_t>(slot)).second) {
            failed = true;
            return std::unexpected(operation_error{"duplicate sensor GPU tile metadata"});
        }
        indices.push_back(coordinate);
    }
    result.candidate_voxels = std::uint64_t(counters.count) * tile_voxels;
    result.kernel_launches = 330;
    result.resident_dispatch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}
}
