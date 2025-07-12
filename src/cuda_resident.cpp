#include "cuda_resident_internal.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace covoxslam
{
cuda_resident_context::cuda_resident_context() = default;
cuda_resident_context::~cuda_resident_context() = default;
std::size_t cuda_resident_context::device_bytes() const noexcept { return state->bytes; }
cuda_resident_order cuda_resident_context::fusion_order() const noexcept { return state->options.order; }
cuda_resident_reduction cuda_resident_context::reduction_mode() const noexcept { return state->options.reduction; }
bool cuda_resident_context::uses_step_tasks() const noexcept { return state->options.step_tasks; }
cuda_resident_warp_options cuda_resident_context::warp_configuration() const noexcept { return state->options.warp; }
cuda_free_space_options cuda_resident_context::free_space_configuration() const noexcept { return state->options.free_space; }
float cuda_resident_context::segment_length() const noexcept { return state->options.segment_length; }
bool cuda_resident_context::uses_compact() const noexcept { return state->options.compact; }
bool cuda_resident_context::uses_compact_batched() const noexcept { return state->options.compact_batched; }
bool cuda_resident_context::uses_fp32() const noexcept { return state->options.fp32; }


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
        if (status) status = data.harvest(slot);
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
    if (data.options.compact || data.options.fp32) {
        if (status) status = data.integrate_compact(input.size(), parameters, slot);
    }
    else if (data.options.step_tasks) {
        if (status) status = data.integrate_tasks(input.size(), parameters, slot);
    }
    else if (data.options.reduction != cuda_resident_reduction::none)
    {
        if (status) status = data.integrate_reduction(input.size(), parameters);
    }
    else {
        if (status) status = data.check(detail::cuda_begin_resident_batch(data.view, data.stream));

        for (unsigned stage = 0; stage < 3 && status; ++stage)
        {
            if (stage == 2) status = data.check(detail::cuda_scan_counts(data.view.counts, data.view.offsets,
                data.view.block_sums, static_cast<std::size_t>(data.options.tile_capacity) * 512, data.stream));

            if (status) status = data.check(detail::cuda_walk_resident_rays(data.view, data.rays, input.size(), parameters, stage, data.stream));
        }
        const auto finish = data.options.order == cuda_resident_order::ordered ? detail::cuda_finish_resident_batch :
             detail::cuda_finish_resident_arrival_batch;

        if (status) status = data.check(finish(data.view,
            {config.truncation_distance, config.maximum_weight, config.extraction_distance}, data.stream));
    }

    if (status) status = data.check(cudaEventRecord(slot.complete, data.stream));
    if (!status) return status;
    slot.pending = true;
    ++data.next_scan;
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

    if (!status) return std::unexpected(status.error());
    status = data.check(cudaMemcpyAsync(&completed, data.view.state, sizeof(completed), cudaMemcpyDeviceToHost, data.stream));
    ++data.statistics.explicit_waits;
    const auto drained = data.check(cudaStreamSynchronize(data.stream));

    if (!status) return std::unexpected(status.error());
    if (!drained) return std::unexpected(drained.error());
    if (completed.error || data.failed) {
        data.failed = true;
        return std::unexpected(operation_error{"resident map is not publishable, device status " + std::to_string(completed.error)});
    }

    for (auto& slot : data.staging) {
        status = data.harvest(slot);

        if (!status) return std::unexpected(status.error());
    }
    auto& statistics = data.statistics;
    statistics.batches = completed.batches;
    statistics.skipped_contributions = completed.skipped;
    statistics.maximum_segment_length = completed.maximum_segment_length;
    statistics.warp_flush_groups = completed.warp_counts[0];
    statistics.warp_peer_groups = completed.warp_counts[1];
     statistics.warp_cache_hits = completed.warp_counts[2];
    statistics.warp_cache_flushes = completed.warp_counts[3];
    auto& mapping = statistics.mapping;
    mapping.cuda_skipped_contributions = statistics.skipped_contributions;
    mapping.cuda_ray_segment_length_m = data.options.segment_length;
    mapping.cuda_ray_segment_slots = data.segments.slots;
    mapping.cuda_accumulator_bits = data.options.reduction == cuda_resident_reduction::none ? 0 : data.options.fp32 ? 32 : 64;
    mapping.cuda_compact_batched = data.options.compact_batched;

    if (data.options.free_space.enabled) {
        mapping.cuda_free_space_minimum_weight = data.options.free_space.minimum_weight;
        mapping.cuda_free_space_refresh_scans = data.options.free_space.refresh_scans;
    }
    mapping.cuda_warp_flush_groups = statistics.warp_flush_groups;
    mapping.cuda_warp_peer_groups = statistics.warp_peer_groups;
    mapping.cuda_warp_cache_hits = statistics.warp_cache_hits;
    mapping.cuda_warp_cache_flushes = statistics.warp_cache_flushes;

    if (data.options.warp.mode != cuda_resident_warp::none) {
        mapping.cuda_warp_count = data.options.warp.count;
        mapping.cuda_warp_count_effective = ((data.options.warp.count + 3) / 4) * 4;
        mapping.cuda_warp_scattered = data.options.warp.scattered;
        mapping.cuda_warp_cache_entries = data.options.warp.mode == cuda_resident_warp::cache ? data.options.warp.cache_entries : 0;
        mapping.cuda_warp_shared_bytes = mapping.cuda_warp_cache_entries * 24 * 4;
    }
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

operation_result<tsdf_tiles> cuda_resident_context::download()
{
    const auto completed = flush();

    if (!completed) return std::unexpected(completed.error());
    auto& data = *state;
    data.failed = true;
    const auto count = completed->mapping.allocated_tiles;

    if (count > data.options.tile_capacity)
        return std::unexpected(operation_error{"resident tile count exceeds its capacity"});
    std::vector<detail::cuda_tile_coordinate> keys(static_cast<std::size_t>(count));
    std::vector<detail::cuda_tsdf_tile> tiles(keys.size());
    operation_status status;
    if (count) {
        status = data.check(cudaMemcpyAsync(keys.data(), data.view.keys, keys.size() * sizeof(keys[0]), cudaMemcpyDeviceToHost, data.stream));

        if (status) status = data.check(cudaMemcpyAsync(tiles.data(), data.view.tiles, tiles.size() * sizeof(tiles[0]), cudaMemcpyDeviceToHost, data.stream));
        ++data.statistics.explicit_waits;
        const auto drained = data.check(cudaStreamSynchronize(data.stream));

        if (status) status = drained;
    }

    if (!status) return std::unexpected(status.error());
    tsdf_tiles result;
    result.reserve(keys.size());

    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        const auto key = keys[index];
        constexpr int bound = 1 << 20;

        if (key.x < -bound || key.x >= bound || key.y < -bound || key.y >= bound || key.z < -bound || key.z >= bound)
            return std::unexpected(operation_error{"resident tile coordinate is unsupported"});
        tsdf_tile tile;

        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& source = tiles[index].voxels[local];

            if (!std::isfinite(source.distance) || !std::isfinite(source.weight) || source.weight < 0 ||
                source.weight > data.config.maximum_weight || (source.flags & ~3U) ||
                std::abs(source.distance) > data.config.truncation_distance)
                return std::unexpected(operation_error{"invalid resident TSDF voxel"});
             tile.voxels[local] = {source.distance, source.weight, source.sample_count, source.flags};
        }
        if (!result.emplace(int3{key.x, key.y, key.z}, std::move(tile)).second)
            return std::unexpected(operation_error{"duplicate resident tile coordinate"});
    }
    data.statistics.mapping.cuda_downloaded_bytes += keys.size() * sizeof(keys[0]) + tiles.size() * sizeof(tiles[0]);
    data.failed = false;
    return result;
}

operation_status cuda_resident_context::reset()
{
    auto& data = *state;
    data.failed = true;
    auto status = data.check(cudaSetDevice(data.options.device));

    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (status) status = data.check(detail::cuda_initialize_resident_map(data.view, data.stream));

    if (status && data.free_space_flags)
        status = data.check(cudaMemsetAsync(data.free_space_flags, 0, std::size_t(data.options.tile_capacity) * 512, data.stream));
    if (status) status = data.initialize_compact();

    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (!status) return status;

    for (auto& slot : data.staging) slot.pending = false;
    data.next_slot = 0;
    data.next_scan = 0;
    data.statistics = {};
     data.failed = false;
    return {};
}

operation_status cuda_resident_context::restore(const tsdf_tiles& source)
{
    auto& data = *state;

    if (source.size() > data.options.tile_capacity)
        return std::unexpected(operation_error{"restored map exceeds resident tile capacity"});
    std::vector<int3> order;
    order.reserve(source.size());

    for (const auto& [key, tile] : source) order.push_back(key);
    std::sort(order.begin(), order.end(), block_index_less);
    std::vector<detail::cuda_tile_coordinate> keys;
    std::vector<detail::cuda_tsdf_tile> tiles(order.size());
    keys.reserve(order.size());

    for (std::size_t index = 0; index < order.size(); ++index)
    {
        const auto key = order[index];
        constexpr int bound = 1 << 20;

        if (key.x < -bound || key.x >= bound || key.y < -bound || key.y >= bound || key.z < -bound || key.z >= bound)
            return std::unexpected(operation_error{"restored tile coordinate is unsupported"});
        keys.push_back({key.x, key.y, key.z});
        const auto& input = source.find(key)->second;

        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& voxel = input.voxels[local];

            if (!std::isfinite(voxel.distance) || !std::isfinite(voxel.weight) || voxel.weight < 0 ||
                voxel.weight > data.config.maximum_weight || (voxel.flags & ~3U) ||
                std::abs(voxel.distance) > data.config.truncation_distance)
                return std::unexpected(operation_error{"invalid restored TSDF voxel"});
             tiles[index].voxels[local] = {voxel.distance, voxel.weight, voxel.sample_count, voxel.flags};
        }
    }
    auto status = reset();
    if (!status) return status;
    data.failed = true;

    if (!keys.empty()) {
        status = data.check(cudaMemcpyAsync(data.view.keys, keys.data(), keys.size() * sizeof(keys[0]), cudaMemcpyHostToDevice, data.stream));
        if (status) status = data.check(cudaMemcpyAsync(data.view.tiles, tiles.data(), tiles.size() * sizeof(tiles[0]), cudaMemcpyHostToDevice, data.stream));
    }

    if (status) status = data.check(detail::cuda_restore_resident_keys(data.view, static_cast<std::uint32_t>(keys.size()), data.stream));
    const auto drained = data.check(cudaStreamSynchronize(data.stream));

    if (!status) return status;
    if (!drained) return drained;
    data.failed = false;
    data.statistics.mapping.cuda_uploaded_bytes += keys.size() * sizeof(keys[0]) + tiles.size() * sizeof(tiles[0]);
    const auto completed = flush();

    if (!completed) return std::unexpected(completed.error());
    return {};
}
}
