#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/mapping/esdf/band_esdf.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/mapping/esdf/detail/band_esdf_workspace.hpp>
#include <covoxslam/mapping/esdf/detail/band_esdf_generation.hpp>
#include "../../../../../cuda/mapping/esdf/detail/driver.cuh"
#include <chrono>

namespace covoxslam::store
{
template<storage::layout l> operation_result<band_esdf_metrics> generate_band_esdf(
    coordinated_store<l>& maps, std::uint64_t id, band_esdf_options options)
{
    if constexpr (l != storage::layout::pool_soa)
        return std::unexpected(operation_error{"band ESDF requires SoA storage"});
    if (!options.valid()) return std::unexpected(operation_error{"invalid band ESDF options"});
    const auto started = std::chrono::steady_clock::now();
    band_esdf_metrics result;
    const auto checked = maps.derive_gpu(id, [&](auto source, auto source_index, auto target,
        auto target_index, auto stream, std::span<const block_key> keys, memory_budget& budget) -> operation_status
    {
        result.tiles = keys.size();

        if (keys.empty()) return {};
        band_esdf_workspace workspace;
        auto valid = workspace.initialize(keys.size(), options.scratch_bytes, budget, stream);
        if (!valid) return valid;
        const auto allocated = std::chrono::steady_clock::now();
        result.allocation_ms = std::chrono::duration<double, std::milli>(allocated - started).count();
        result.scratch_bytes = workspace.reserved_bytes;
        result.peak_device_bytes = budget.snapshot().peak[2];
        auto descriptor = workspace.fields.descriptor();

        if (!descriptor) return std::unexpected(descriptor.error());
        auto& metadata = workspace.metadata;
        auto host_memory = budget.reserve({keys.size_bytes(), 0, 0});
        if (!host_memory) return std::unexpected(host_memory.error());
        valid = storage::cuda_storage_status(cudaMemcpyAsync(metadata.template get<block_key>(0),
            keys.data(), keys.size_bytes(), cudaMemcpyHostToDevice, stream));
        if (!valid) return valid;
        result.upload_bytes = keys.size_bytes();
        result.upload_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - allocated).count();
        const auto code = run_band_esdf(source.native_accessor().descriptor, source_index,
            target.native_accessor().descriptor, target_index, *descriptor,
            metadata.template get<block_key>(0), unsigned(keys.size()),
            metadata.template get<detail::cuda_tile_coordinate>(1), metadata.template get<void>(2),
            metadata.template get<unsigned>(3), options, &result, stream);
        valid = esdf_launch_status(code);

        if (!valid) return valid;
        return workspace.complete();
    }, false);
    if (!checked) return std::unexpected(checked.error());
    const auto state = maps.status(id);
    if (!state) return std::unexpected(state.error());
    auto current = validate_band_generation(*state);
    if (!current) return std::unexpected(current.error());
    if (state->gpu[1] != state->content[1])
        return std::unexpected(operation_error{"ESDF completion generation is stale"});
    result.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}
}
