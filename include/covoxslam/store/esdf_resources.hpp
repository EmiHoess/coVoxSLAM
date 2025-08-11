#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/store/coordinator.hpp>

namespace covoxslam::store
{
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
}
