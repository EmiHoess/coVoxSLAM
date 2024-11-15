#include "cuda_resident_internal.hpp"
#include <bit>
#include <chrono>

namespace covoxslam
{
cuda_resident_context::cuda_resident_context() = default;
cuda_resident_context::~cuda_resident_context() = default;
std::size_t cuda_resident_context::device_bytes() const noexcept { return state->bytes; }

operation_result<std::unique_ptr<cuda_resident_context>> cuda_resident_context::create(
    mapping_config config, cuda_resident_options options)
{

    if (!config.valid() || !options.tile_capacity || options.tile_capacity > UINT32_MAX / 512 ||
        options.tile_capacity > config.maximum_tiles || !options.ray_capacity || !options.contribution_capacity)
        return std::unexpected(operation_error{"invalid resident mapping configuration or capacities"});
    const auto voxels = static_cast<std::size_t>(options.tile_capacity) * 512;
    const auto buckets = std::bit_ceil(options.tile_capacity * 2U);
    const auto sums = (voxels + 255) / 256;
    const auto bytes = sizeof(detail::cuda_resident_state) + std::size_t(buckets) * sizeof(detail::cuda_resident_hash_entry) +
        options.tile_capacity * sizeof(detail::cuda_tile_coordinate) + voxels * (2 * sizeof(detail::cuda_tsdf_voxel) + 16) +
        sums * sizeof(std::uint64_t) + options.contribution_capacity * sizeof(detail::cuda_resident_contribution) +
        options.ray_capacity * sizeof(detail::cuda_device_ray);

    if (options.maximum_device_bytes && bytes > options.maximum_device_bytes)
        return std::unexpected(operation_error{"resident storage exceeds the device byte budget"});
    const auto device = query_cuda_device_checked(options.device);
    if (!device) return std::unexpected(device.error());

    if (bytes > device->total_memory_bytes)
        return std::unexpected(operation_error{"resident storage exceeds total device memory"});
    auto owner = std::unique_ptr<cuda_resident_context>(new cuda_resident_context);
    owner->state = std::make_unique<implementation>();
    auto& data = *owner->state;
    data.config = config;
     data.options = options;
    data.bytes = bytes;
    auto status = data.check(cudaSetDevice(options.device));

    if (!status) return std::unexpected(status.error());
    status = data.check(cudaStreamCreateWithFlags(&data.stream, cudaStreamNonBlocking));
    if (!status) return std::unexpected(status.error());
    auto& view = data.view;
    view.tile_capacity = options.tile_capacity;
    view.hash_capacity = buckets;
    view.contribution_capacity = options.contribution_capacity;
    status = data.allocate(view.state, 1);

    if (status) status = data.allocate(view.hash, buckets);
    if (status) status = data.allocate(view.keys, options.tile_capacity);
    if (status) status = data.allocate(view.tiles, options.tile_capacity);

    if (status) status = data.allocate(view.pending, voxels);
    if (status) status = data.allocate(view.counts, voxels);

    if (status) status = data.allocate(view.cursors, voxels);
    if (status) status = data.allocate(view.offsets, voxels);

    if (status) status = data.allocate(view.block_sums, sums);
    if (status) status = data.allocate(view.contributions, options.contribution_capacity);

    if (status) status = data.allocate(data.rays, options.ray_capacity);

    for (auto& slot : data.staging)
    {

        if (!status) break;
        status = data.check(cudaHostAlloc(reinterpret_cast<void**>(&slot.rays),
            options.ray_capacity * sizeof(detail::cuda_device_ray), cudaHostAllocDefault));
        if (status) status = data.check(cudaEventCreateWithFlags(&slot.complete, cudaEventDisableTiming));
    }
    if (status) status = data.check(detail::cuda_initialize_resident_map(view, data.stream));

    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (!status) return std::unexpected(status.error());
    return owner;
}

operation_status cuda_resident_context::submit(std::span<const ray> input)
{
    auto& data = *state;

    if (data.failed) return std::unexpected(operation_error{"resident mapping is failed until reset"});
    if (input.size() > data.options.ray_capacity) {
        data.failed = true;
        return std::unexpected(operation_error{"resident scan exceeds ray capacity"});
    }
    const auto start = std::chrono::steady_clock::now();
    auto status = data.check(cudaSetDevice(data.options.device));
    auto& slot = data.staging[data.next_slot];

    if (status && slot.pending) {
        ++data.statistics.explicit_waits;
        status = data.check(cudaEventSynchronize(slot.complete));
    }
    if (!status) return status;

    for (std::size_t index = 0; index < input.size(); ++index)
    {
        const auto& ray = input[index];

        if (ray.frame_id < 0 || ray.column > UINT32_MAX || ray.ring > UINT32_MAX) {
            data.failed = true;
            return std::unexpected(operation_error{"resident ray metadata exceeds its supported range"});
        }
        slot.rays[index] = {{ray.origin.x, ray.origin.y, ray.origin.z}, {ray.endpoint.x, ray.endpoint.y, ray.endpoint.z},
            ray.weight, ray.stamp_ns, static_cast<std::uint32_t>(ray.frame_id), static_cast<std::uint32_t>(ray.column),
            static_cast<std::uint32_t>(ray.ring), static_cast<std::uint32_t>(index)};
    }
    const auto bytes = input.size_bytes() / sizeof(ray) * sizeof(detail::cuda_device_ray);
    const auto& config = data.config;
    const detail::cuda_ray_parameters parameters{config.voxel_size, config.truncation_distance,
        config.minimum_ray_length_m, config.maximum_ray_length_m, data.options.voxel_carving, data.options.allow_clear};

    if (bytes) status = data.check(cudaMemcpyAsync(data.rays, slot.rays, bytes, cudaMemcpyHostToDevice, data.stream));
    if (status) status = data.check(detail::cuda_begin_resident_batch(data.view, data.stream));

    for (unsigned stage = 0; stage < 3 && status; ++stage)
    {
        if (stage == 2) status = data.check(detail::cuda_scan_counts(data.view.counts, data.view.offsets,
            data.view.block_sums, static_cast<std::size_t>(data.options.tile_capacity) * 512, data.stream));

        if (status) status = data.check(detail::cuda_walk_resident_rays(data.view, data.rays, input.size(), parameters, stage, data.stream));
    }
    if (status) status = data.check(detail::cuda_finish_resident_batch(data.view,
        {config.truncation_distance, config.maximum_weight, config.extraction_distance}, data.stream));

    if (status) status = data.check(cudaEventRecord(slot.complete, data.stream));
    if (!status) return status;
    slot.pending = true;
    data.next_slot = (data.next_slot + 1) % data.staging.size();
     data.statistics.mapping.rays_submitted += input.size();
    data.statistics.mapping.cuda_uploaded_bytes += bytes;
    data.statistics.mapping.integration_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return {};
}

operation_result<cuda_resident_statistics> cuda_resident_context::flush()
{
    auto& data = *state;
    const auto start = std::chrono::steady_clock::now();
    auto status = data.check(cudaSetDevice(data.options.device));
    detail::cuda_resident_state completed{};

    if (status) status = data.check(cudaMemcpyAsync(&completed, data.view.state, sizeof(completed), cudaMemcpyDeviceToHost, data.stream));
    if (status) {
        ++data.statistics.explicit_waits;
        status = data.check(cudaStreamSynchronize(data.stream));
    }

    if (!status) return std::unexpected(status.error());
    if (completed.error || data.failed) {
        data.failed = true;
        return std::unexpected(operation_error{"resident map is not publishable, device status " + std::to_string(completed.error)});
    }

    for (auto& slot : data.staging) slot.pending = false;
    auto& statistics = data.statistics;
    statistics.batches = completed.batches;
    statistics.maximum_segment_length = completed.maximum_segment_length;
    statistics.mapping.rays_integrated = statistics.mapping.cuda_contributing_rays = completed.integrated;
    statistics.mapping.rays_rejected = completed.rejected;
     statistics.mapping.voxel_updates = completed.contributions;
    statistics.mapping.cuda_voxel_segments = completed.touched;
    statistics.mapping.allocated_tiles = completed.tile_count;
    statistics.mapping.cuda_downloaded_bytes += sizeof(completed);
    statistics.mapping.integration_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return statistics;
}
}
