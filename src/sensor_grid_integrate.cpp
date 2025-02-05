#include "sensor_grid_map_internal.hpp"
#include <chrono>
#include <cmath>

namespace covoxslam::sensor_grid
{
operation_result<transfer_times> gpu_map::integrate(const gpu_grid& source, pose transform, interpolation mode)
{
    auto& data = *state;
    const auto field = source.device_field();
    const auto q = transform.rotation;
    const bool masked = candidate_uses_mask(data.options.candidates);
    const bool coarse = data.options.candidates == candidate_mode::coarse_mask;
    const bool listed = candidate_uses_list(data.options.candidates);
    const bool append = data.options.candidates == candidate_mode::append_fp32;
    const double norm = double(q.x) * q.x + double(q.y) * q.y + double(q.z) * q.z + double(q.w) * q.w;

    if (data.failed || !field.voxels || source.device_index() != data.device || unsigned(mode) > 5 ||
        (masked && unsigned(mode) > 2) ||
        (data.options.arithmetic == transfer_arithmetic::fp32 && unsigned(mode) > 2) ||
        (data.options.specialized && (data.options.arithmetic != transfer_arithmetic::fp64 || mode != interpolation::trilinear)) ||
        field.side != data.grid.side || field.origin != data.grid.origin || field.voxel_size != data.settings.voxel_size ||
        !std::isfinite(norm) || std::abs(norm - 1) > .001 || !std::isfinite(transform.translation.x) ||
        !std::isfinite(transform.translation.y) || !std::isfinite(transform.translation.z))
        return std::unexpected(operation_error{"invalid sensor TSDF transfer"});
    transform.rotation = normalized(q);
    const auto rotation = transform.rotation;
    const detail::cuda_projective_pose packed{{transform.translation.x, transform.translation.y, transform.translation.z},
        {rotation.x, rotation.y, rotation.z, rotation.w}};
    const bool gpu_discovery = data.options.discovery == discovery_mode::gpu_host;
    auto status = data.check(cudaSetDevice(data.device));
    if (status) status = data.check(cudaEventRecord(data.events[0], data.stream));
    if (status) status = data.check(data.options.source_mask == source_mask_mode::compact
        ? cuda_sensor_compact_mask(field, data.mask_offsets, data.mask_cells, data.mask, data.stream)
        : cuda_sensor_mask(field, data.mask, data.stream));
    if (status) status = data.check(cudaEventRecord(data.events[1], data.stream));
    if (status && !gpu_discovery) status = data.check(cudaMemcpyAsync(data.flags.data(), data.mask, data.flags.size(), cudaMemcpyDeviceToHost, data.stream));
    if (!gpu_discovery || !status) status = data.wait(status);
    if (!status) return std::unexpected(status.error());
    const auto start = std::chrono::steady_clock::now();
    transfer_times result;
    auto candidates = gpu_discovery ? data.discover_gpu(field, packed, result.gpu_discovery_ms) :
        destination_tiles(data.grid, data.settings.voxel_size, data.flags, transform, data.capacity);
    if (!candidates) return std::unexpected(candidates.error());
    std::size_t additional{};

    for (const auto key : *candidates) additional += !data.index.contains(key);
    if (additional > data.capacity - data.indices.size())
        return std::unexpected(operation_error{"sensor TSDF tile capacity exhausted"});
    const auto first = data.indices.size();
    std::vector<std::uint32_t> slots;
    std::vector<detail::cuda_tile_coordinate> new_keys;
    slots.reserve(candidates->size());
     new_keys.reserve(additional);

    for (const auto key : *candidates)
    {
        const auto [found, inserted] = data.index.emplace(key, static_cast<std::uint32_t>(data.indices.size()));

        if (inserted) {
            data.indices.push_back(key);
             new_keys.push_back({key.x, key.y, key.z});
        }
        slots.push_back(found->second);
    }
    result.candidate_voxels = std::uint64_t(slots.size()) * 512;
    result.kernel_launches = 1 + !slots.empty() + gpu_discovery;
    if (listed && !slots.empty()) ++result.kernel_launches;
    result.discovery_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (!new_keys.empty()) status = data.check(cudaMemcpyAsync(data.keys + first, new_keys.data(),
        new_keys.size() * sizeof(*data.keys), cudaMemcpyHostToDevice, data.stream));
    if (status && !slots.empty()) status = data.check(cudaMemcpyAsync(data.slots, slots.data(), slots.size() * 4, cudaMemcpyHostToDevice, data.stream));
    if (status) status = data.check(cudaMemsetAsync(data.updates, 0, 8, data.stream));
    if (status && masked) status = data.check(cudaMemsetAsync(data.candidate_count, 0, 8, data.stream));
    if (status) status = data.check(cudaEventRecord(data.events[4], data.stream));
    unsigned marking_launches{};
    const auto mark_list = data.options.candidates == candidate_mode::append_fp64 ? cuda_sensor_append_candidates_fp64 :
        append ? cuda_sensor_append_candidates :
        data.options.candidates == candidate_mode::compact_fp64 ? cuda_sensor_list_candidates_fp64 : cuda_sensor_list_candidates;
    if (status && listed) status = data.check(mark_list(field, mode, data.active_cells, data.active_count,
        data.slots, static_cast<std::uint32_t>(slots.size()), data.keys, packed, data.candidate_list, data.stream, &marking_launches));
    const auto mark_mask = coarse ? cuda_sensor_coarse_candidates : cuda_sensor_candidates;
    if (status && masked && !listed) status = data.check(mark_mask(field, mode, data.active_cells, data.active_count,
        data.slots, static_cast<std::uint32_t>(slots.size()), data.keys, packed, data.candidate_masks, data.stream, &marking_launches));
    if (status) status = data.check(cudaEventRecord(data.events[5], data.stream));
    if (status) status = data.check(cudaEventRecord(data.events[2], data.stream));
    if (status && listed) status = data.check(cuda_sensor_list_transfer(field, mode, data.slots,
        static_cast<std::uint32_t>(slots.size()), data.keys, data.tiles, packed, data.settings.voxel_size,
        data.settings.maximum_weight, data.settings.voxel_size, data.candidate_list, data.updates, data.candidate_count, data.stream));
    if (status && coarse) status = data.check(cuda_sensor_coarse_transfer(field, mode, data.slots,
        static_cast<std::uint32_t>(slots.size()), data.keys, data.tiles, packed, data.settings.voxel_size,
        data.settings.maximum_weight, data.settings.voxel_size, data.candidate_masks, data.updates, data.candidate_count, data.stream));
    if (status && masked && !listed && !coarse) status = data.check(cuda_sensor_masked_transfer(field, mode, data.slots,
        static_cast<std::uint32_t>(slots.size()), data.keys, data.tiles, packed, data.settings.voxel_size,
        data.settings.maximum_weight, data.settings.voxel_size, data.candidate_masks, data.updates,
        data.candidate_count, data.stream, data.options.candidates == candidate_mode::compacted_control));
    if (status && !masked) status = data.check(cuda_sensor_transfer(field, mode, data.slots, static_cast<std::uint32_t>(slots.size()),
        data.keys, data.tiles, packed, data.settings.voxel_size, data.settings.maximum_weight,
        data.settings.voxel_size, data.updates, data.stream, data.options.arithmetic == transfer_arithmetic::fp32,
        data.options.specialized, data.settings.truncation));
    if (status) status = data.check(cudaEventRecord(data.events[3], data.stream));
    if (status) status = data.check(cudaMemcpyAsync(&result.updates, data.updates, 8, cudaMemcpyDeviceToHost, data.stream));
    if (status && masked) status = data.check(cudaMemcpyAsync(&result.candidate_voxels, data.candidate_count, 8, cudaMemcpyDeviceToHost, data.stream));
    candidate_list_counters list_counts;
    if (status && listed) status = data.check(cudaMemcpyAsync(&list_counts, data.candidate_list.counters,
        sizeof(list_counts), cudaMemcpyDeviceToHost, data.stream));
    status = data.wait(status);
    if (status) status = data.check(cudaEventElapsedTime(&result.mask_ms, data.events[0], data.events[1]));
    if (status) status = data.check(cudaEventElapsedTime(&result.fusion_ms, data.events[2], data.events[3]));
    if (status && masked) status = data.check(cudaEventElapsedTime(&result.candidates_ms, data.events[4], data.events[5]));
    if (!status) return std::unexpected(status.error());
    result.candidate_overflows = list_counts.overflow != 0;
    result.kernel_launches += marking_launches;
    if (list_counts.invalid) {
        data.failed = true;
        return std::unexpected(operation_error{"candidate list violates unique tile ownership"});
    }
    return result;
}
}
