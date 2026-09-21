#include "gpu_band_esdf.hpp"
#include "mapping/detail/types.cuh"
#include "mapping/esdf/detail/driver.cuh"
#include <covoxslam/storage/device_buffer.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <cuda_runtime_api.h>

namespace covoxslam::store
{
namespace
{
operation_status validate_band_generation(const authority_snapshot& state)
{
    auto checked = require_current_esdf(state, 2);

    if (!checked) return checked;
    if (state.state == lifecycle::retired || state.writer != writer_kind::none)
        return std::unexpected(operation_error{"ESDF generation is not available"});
    return {};
}
struct esdf_scratch
{
    std::shared_ptr<budget_claim> memory;
    block_key* keys{};
    detail::cuda_tile_coordinate* coordinates{};
    band_tile_slot* slots{};
    band_status* status{};
    cudaStream_t stream{};
    int device{-1};
    esdf_scratch() = default;
    esdf_scratch(const esdf_scratch&) = delete;
    esdf_scratch& operator=(const esdf_scratch&) = delete;
    ~esdf_scratch()
    {
        if (device < 0) return;
        int previous{-1};
        const auto queried = cudaGetDevice(&previous);
        if (cudaSetDevice(device) != cudaSuccess) return;
        cudaStreamSynchronize(stream);

        if (keys) cudaFree(keys);
        if (coordinates) cudaFree(coordinates);
        if (slots) cudaFree(slots);
        if (status) cudaFree(status);
        if (queried == cudaSuccess && previous != device) cudaSetDevice(previous);
    }
    operation_status initialize(std::size_t count, std::size_t maximum,
        memory_budget& budget, cudaStream_t selected)
    {
        if (memory || keys || coordinates || slots || status || !count)
            return std::unexpected(operation_error{"ESDF scratch is invalid"});
        constexpr auto per_tile = sizeof(block_key) + sizeof(detail::cuda_tile_coordinate) +
            sizeof(band_tile_slot);
        constexpr auto fixed = sizeof(band_status);
        if (maximum < fixed || count > (maximum - fixed) / per_tile)
            return std::unexpected(operation_error{"ESDF scratch limit exceeded"});
        const auto total = count * per_tile + fixed;
        auto claimed = budget.reserve({0, 0, total});
        if (!claimed) return std::unexpected(claimed.error());
        memory = std::move(*claimed); stream = selected;
        auto checked = storage::cuda_storage_status(cudaGetDevice(&device));
        if (!checked) return checked;
        checked = storage::cuda_storage_status(
            cudaMalloc(reinterpret_cast<void**>(&keys), count * sizeof(*keys)));
        if (!checked) return checked;
        checked = storage::cuda_storage_status(
            cudaMalloc(reinterpret_cast<void**>(&coordinates), count * sizeof(*coordinates)));
        if (!checked) return checked;
        checked = storage::cuda_storage_status(
            cudaMalloc(reinterpret_cast<void**>(&slots), count * sizeof(*slots)));
        if (!checked) return checked;
        checked = storage::cuda_storage_status(
            cudaMalloc(reinterpret_cast<void**>(&status), sizeof(*status)));
        if (!checked) return checked;
        return {};
    }
};
operation_status esdf_launch_status(int error)
{
    return storage::cuda_storage_status(static_cast<cudaError_t>(error));
}
operation_result<std::size_t> band_workspace_bytes(std::size_t count, std::size_t maximum)
{
    constexpr auto field_bytes = storage::payload_tile_bytes<storage::schema::projective_esdf,
        storage::layout::pool_soa>;
    constexpr auto per_tile = field_bytes + sizeof(block_key) +
        sizeof(detail::cuda_tile_coordinate) + sizeof(band_tile_slot);
    constexpr auto fixed = sizeof(band_status);

    if (!count || count > UINT32_MAX / tile_voxels || maximum < fixed ||
        count > (maximum - fixed) / per_tile)
        return std::unexpected(operation_error{"band ESDF workspace exceeds its limit"});
    return count * per_tile + fixed;
}
struct band_esdf_workspace
{
    std::shared_ptr<budget_claim> field_memory;
    storage::device_buffer<storage::schema::projective_esdf, storage::layout::pool_soa> fields;
    esdf_scratch metadata;
     std::size_t reserved_bytes{};
    band_esdf_workspace() = default;
    band_esdf_workspace(const band_esdf_workspace&) = delete;
    band_esdf_workspace& operator=(const band_esdf_workspace&) = delete;
    operation_status complete()
    {
        return storage::cuda_storage_status(cudaStreamSynchronize(metadata.stream));
    }
    operation_status initialize(std::size_t count, std::size_t maximum,
        memory_budget& budget, cudaStream_t stream)
    {
        if (field_memory) return std::unexpected(operation_error{"band workspace is already allocated"});
        auto total = band_workspace_bytes(count, maximum);
        if (!total) return std::unexpected(total.error());
        constexpr auto field_bytes = storage::payload_tile_bytes<storage::schema::projective_esdf,
            storage::layout::pool_soa>;
        const auto bytes = count * field_bytes;
        auto claimed = budget.reserve({0, 0, bytes});

        if (!claimed) return std::unexpected(claimed.error());
        field_memory = std::move(*claimed);
        auto checked = fields.initialize(count);

        if (!checked) return checked;
        checked = metadata.initialize(count, maximum - bytes, budget, stream);
        if (!checked) return checked;
        reserved_bytes = *total;
        return {};
    }
};
}

operation_result<band_esdf_metrics> generate_band_esdf(
    coordinated_store<storage::layout::pool_soa>& maps, std::uint64_t id, band_esdf_options options)
{
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
        valid = storage::cuda_storage_status(cudaMemcpyAsync(metadata.keys,
            keys.data(), keys.size_bytes(), cudaMemcpyHostToDevice, stream));
        if (!valid) return valid;
        result.upload_bytes = keys.size_bytes();
        result.upload_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - allocated).count();
        const auto code = run_band_esdf(source.native_accessor().descriptor, source_index,
            target.native_accessor().descriptor, target_index, *descriptor,
            metadata.keys, unsigned(keys.size()), metadata.coordinates, metadata.slots,
            metadata.status, options, &result, stream);
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
