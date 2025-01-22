#include <covoxslam/sensor_grid_combined.hpp>
#include <chrono>

namespace covoxslam::sensor_grid
{
operation_result<std::unique_ptr<combined_map>> combined_map::create(const lookup& table,
    std::uint32_t capacity, int device, std::size_t maximum_bytes, gather_options options, transfer_options transfer)
{
    if (!combined_gather(options.backend))
        return std::unexpected(operation_error{"combined mapping requires combined gathering"});
    auto grid = gpu_grid::create(table, device, maximum_bytes, options);

    if (!grid) return std::unexpected(grid.error());
    auto map = gpu_map::create(table, capacity, device, maximum_bytes - (*grid)->device_bytes(), transfer);

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
    totals.transfer.fusion_ms += transferred->fusion_ms;
    totals.transfer.updates += transferred->updates;
    totals.transfer.candidate_voxels += transferred->candidate_voxels;
    totals.transfer.kernel_launches += transferred->kernel_launches;
    totals.kernel_launches += result.kernel_launches;
     return result;
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
