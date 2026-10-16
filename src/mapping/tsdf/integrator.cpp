#include <covoxslam/mapping/tsdf/integrator.hpp>
#include <covoxslam/sensor/detail/precision.hpp>
#include <chrono>

namespace covoxslam::sensor_grid
{
config sensor_config_from_mapping(const mapping_config& mapping) noexcept
{
    config result;
    result.voxel_size = mapping.voxel_size;
    result.radius = mapping.maximum_ray_length_m;
    result.truncation = mapping.truncation_distance;
    result.minimum_range = mapping.minimum_ray_length_m;
    result.maximum_weight = mapping.maximum_weight;
    return result;
}

mapping_preset combined_nearest_preset(const mapping_config& mapping) noexcept
{
    mapping_preset result;
    result.mapping = mapping;
    result.gather = {gather_backend::hybrid_combined, combined_short_list_max,
        combined_block_list_min, execution_mode::stream};
    result.transfer = {};
    result.sampling = interpolation::nearest;
    result.ring_stride = result.column_stride = 1;
    return result;
}

bool matches_combined_nearest(const mapping_preset& options) noexcept
{
    const auto preset = combined_nearest_preset(mapping_config{});
    return accumulation_bits == 32 && options.sampling == preset.sampling &&
        options.ring_stride == preset.ring_stride && options.column_stride == preset.column_stride &&
        options.mapping.voxel_size == preset.mapping.voxel_size &&
        options.mapping.truncation_distance == preset.mapping.truncation_distance &&
        options.mapping.maximum_ray_length_m == preset.mapping.maximum_ray_length_m &&
        options.mapping.maximum_weight == preset.mapping.maximum_weight &&
        options.gather.backend == preset.gather.backend &&
        options.gather.short_list_max == preset.gather.short_list_max &&
        options.gather.block_list_min == preset.gather.block_list_min &&
        options.gather.execution == preset.gather.execution &&
        options.transfer.discovery == preset.transfer.discovery &&
        options.transfer.arithmetic == preset.transfer.arithmetic &&
        options.transfer.candidates == preset.transfer.candidates &&
        options.transfer.source_mask == preset.transfer.source_mask &&
        options.transfer.specialized == preset.transfer.specialized;
}
operation_result<mapping_preset> combined_nearest_preset()
{
    if constexpr (accumulation_bits != 32)
        return std::unexpected(operation_error{"combined nearest requires an FP32 sensor accumulation build"});
    return combined_nearest_preset(mapping_config{});
}
operation_result<std::unique_ptr<combined_map>> combined_map::create(const lookup& table,
    std::uint32_t capacity, int device, std::size_t maximum_bytes, gather_options options, transfer_options transfer,
    bool resident_soa)
{
    if (options.backend != gather_backend::hybrid_combined || !resident_soa)
        return std::unexpected(operation_error{"coVoxSLAM requires combined gathering and resident SoA storage"});
    auto grid = gpu_grid::create(table, device, maximum_bytes, options);

    if (!grid) return std::unexpected(grid.error());
    auto map = gpu_map::create(table, capacity, device, maximum_bytes - (*grid)->device_bytes(), transfer, resident_soa);

    if (!map) return std::unexpected(map.error());
    auto result = std::unique_ptr<combined_map>(new combined_map);
    result->grid = std::move(*grid);
     result->map = std::move(*map);
    return result;
}
operation_result<combined_times> combined_map::integrate(std::span<const float> ranges,
    pose transform, interpolation mode, gather_policy policy)
{
    if (failed || totals.completed_scans == UINT64_MAX)
        return std::unexpected(operation_error{"combined mapper cannot accept another scan"});
    const auto start = std::chrono::steady_clock::now();
    auto gathered = grid->update(ranges, policy);

    if (!gathered) { failed = true; return std::unexpected(gathered.error()); }
    auto transferred = map->integrate(*grid, transform, mode);

    if (!transferred) { failed = true; return std::unexpected(transferred.error()); }
    combined_times result;
    result.gather = *gathered;
    result.transfer = *transferred;
    const auto rows = grid->statistics();
    result.kernel_launches = 1 + transferred->kernel_launches;
    if (policy.first_entry_only) ++result.kernel_launches;
    else {
        result.kernel_launches += rows.short_rows != 0;
        result.kernel_launches += policy.max_rays ? rows.warp_rows + rows.block_rows != 0 : rows.warp_rows != 0;
        result.kernel_launches += !policy.max_rays && rows.block_rows != 0;
    }
    result.integration_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    result.completed_scans = ++totals.completed_scans;
    totals.integration_ms += result.integration_ms;
    totals.gather.upload_ms += gathered->upload_ms;
    totals.gather.gather_ms += gathered->gather_ms;
    totals.gather.graph_setup_ms += gathered->graph_setup_ms;
    totals.transfer.mask_ms += transferred->mask_ms;
    totals.transfer.discovery_ms += transferred->discovery_ms;
    totals.transfer.gpu_discovery_ms += transferred->gpu_discovery_ms;
    totals.transfer.gpu_allocation_ms += transferred->gpu_allocation_ms;
    totals.transfer.resident_dispatch_ms += transferred->resident_dispatch_ms;
    totals.transfer.fusion_ms += transferred->fusion_ms;
    totals.transfer.candidates_ms += transferred->candidates_ms;
    totals.transfer.updates += transferred->updates;
    totals.transfer.candidate_voxels += transferred->candidate_voxels;
    totals.transfer.candidate_overflows += transferred->candidate_overflows;
    totals.transfer.kernel_launches += transferred->kernel_launches;
    totals.kernel_launches += result.kernel_launches;
     return result;
}
operation_result<std::size_t> combined_map::retain(
    store::coordinated_store<storage::layout::pool_soa, storage::schema::esdf>& destination,
    std::uint64_t id, pose4 anchor)
{
    if (failed) return std::unexpected(operation_error{"cannot retain failed combined mapping"});
    return map->retain(destination, id, anchor);
}
operation_result<tsdf_tiles> combined_map::download()
{
    if (failed) return std::unexpected(operation_error{"cannot publish failed combined mapping"});
    const auto start = std::chrono::steady_clock::now();
    auto result = map->download();

    if (!result) { failed = true; return result; }
    totals.readback_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    return result;
}
operation_status combined_map::reset()
{
    if (failed) return std::unexpected(operation_error{"cannot reset failed combined mapping"});
    const auto result = map->reset();

    if (!result) { failed = true; return result; }
    totals = {};
    return {};
}
}
