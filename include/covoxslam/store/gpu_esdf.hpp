#pragma once
#include <covoxslam/store/coordinator.hpp>
#include "../../../cuda/store/mapping.h"

namespace covoxslam::store
{
struct esdf_execution_limits
{
    std::size_t scratch_bytes{256U * 1024U * 1024U};
    std::uint32_t maximum_rounds{10000}, pops_per_launch{4096};
    std::uint64_t maximum_pops{100000000};
};
struct esdf_scratch
{
    std::shared_ptr<budget_claim> memory;
    std::array<void*, 12> buffers{};
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

        for (auto pointer : buffers) if (pointer) cudaFree(pointer);
        if (queried == cudaSuccess && previous != device) cudaSetDevice(previous);
    }
    operation_status initialize(std::span<const std::size_t> sizes, std::size_t maximum,
        memory_budget& budget, cudaStream_t selected)
    {
        if (memory || sizes.size() > buffers.size()) return std::unexpected(operation_error{"ESDF scratch is invalid"});
        std::size_t total{};

        for (auto bytes : sizes) {
            if (bytes > maximum - total) return std::unexpected(operation_error{"ESDF scratch limit exceeded"});
            total += bytes;
        }
        auto claimed = budget.reserve({0, 0, total});
        if (!claimed) return std::unexpected(claimed.error());
        memory = std::move(*claimed); stream = selected;
        auto checked = storage::cuda_storage_status(cudaGetDevice(&device));
        if (!checked) return checked;

        for (std::size_t i = 0; i < sizes.size(); ++i)
            if (sizes[i]) {
                checked = storage::cuda_storage_status(cudaMalloc(&buffers[i], sizes[i]));
                if (!checked) return checked;
            }
        return {};
    }
    template<class t> t* get(std::size_t i) const noexcept { return static_cast<t*>(buffers[i]); }
    template<class t> operation_status read(t& value, const t* device_value) const
    {
        auto checked = storage::cuda_storage_status(cudaMemcpyAsync(&value, device_value, sizeof(t), cudaMemcpyDeviceToHost, stream));
        const auto waited = storage::cuda_storage_status(cudaStreamSynchronize(stream));
        return checked ? waited : checked;
    }
    operation_status check(const unsigned* status) const
    {
        unsigned value{};
        auto checked = read(value, status);
        if (!checked) return checked;
        if (value) return std::unexpected(operation_error{"ESDF device validation failed"});
        return {};
    }
};
inline operation_status esdf_launch_status(int error)
{
    return storage::cuda_storage_status(static_cast<cudaError_t>(error));
}
template<storage::layout l> operation_status generate_site_esdf(
    coordinated_store<l, storage::schema::projective_esdf>& maps, std::uint64_t id,
    detail::cuda_site_esdf_parameters parameters, esdf_execution_limits limits = {})
{
    auto validated = esdf_launch_status(detail::cuda_initialize_site_esdf(nullptr, nullptr, 0, parameters, nullptr, nullptr, nullptr));
    if (!validated) return validated;
    if (!limits.maximum_rounds) return std::unexpected(operation_error{"ESDF iteration limit is zero"});
    return maps.derive_gpu(id, [&](auto source, auto source_index, auto target, auto target_index, auto stream,
        std::span<const block_key> keys, memory_budget& budget) -> operation_status {
        const auto count = keys.size();
        if (!count || count > UINT32_MAX / 512) return std::unexpected(operation_error{"ESDF topology is empty or oversized"});
        const std::array<std::size_t, 5> sizes{keys.size_bytes(), count * sizeof(detail::cuda_tile_coordinate),
            count * sizeof(detail::cuda_tsdf_tile), count * 512 * sizeof(detail::cuda_site_esdf_voxel), 8};
        esdf_scratch scratch;
        auto checked = scratch.initialize(sizes, limits.scratch_bytes, budget, stream);
        if (!checked) return checked;
        auto* device_keys = scratch.get<block_key>(0);
        auto* ordered_keys = scratch.get<detail::cuda_tile_coordinate>(1);
        auto* input = scratch.get<detail::cuda_tsdf_tile>(2);
        auto* output = scratch.get<detail::cuda_site_esdf_voxel>(3);
        auto* flags = scratch.get<unsigned>(4);
        checked = storage::cuda_storage_status(cudaMemcpyAsync(device_keys, keys.data(), keys.size_bytes(), cudaMemcpyHostToDevice, stream));
        if (!checked) return checked;
        checked = esdf_launch_status(pack_esdf_input(source.native_accessor().descriptor, source_index, device_keys,
            unsigned(count), ordered_keys, input, flags, stream));
        if (!checked) return checked;
        checked = scratch.check(flags);
        if (!checked) return checked;
        checked = esdf_launch_status(detail::cuda_initialize_site_esdf(ordered_keys, input, count, parameters, output, flags, stream));
        if (!checked) return checked;
        checked = scratch.check(flags);
        if (!checked) return checked;
        const float distance = parameters.maximum_distance_m / parameters.voxel_size;
        unsigned changed{1}, rounds{};
        while (changed && rounds < limits.maximum_rounds)
        {
            checked = storage::cuda_storage_status(cudaMemsetAsync(flags + 1, 0, 4, stream));
            if (!checked) return checked;
            checked = esdf_launch_status(detail::cuda_sweep_site_esdf(ordered_keys, count, output, distance * distance, flags + 1, stream));
            if (!checked) return checked;
            checked = esdf_launch_status(detail::cuda_relax_site_esdf_neighbors(ordered_keys, count, output, distance * distance, flags + 1, stream));
            if (!checked) return checked;
            checked = scratch.read(changed, flags + 1);
            if (!checked) return checked;
            ++rounds;
        }
        if (changed) return std::unexpected(operation_error{"ESDF iteration limit exceeded"});
        checked = esdf_launch_status(unpack_esdf_output(target.native_accessor().descriptor, target_index, device_keys,
            unsigned(count), output, storage::schema::projective_esdf, flags, stream));
        if (!checked) return checked;
        return scratch.check(flags);
    });
}
}
