#pragma once
#include <covoxslam/storage/draft.hpp>
#include <covoxslam/storage/device_view.hpp>
#include <cuda_runtime_api.h>

namespace covoxslam::storage
{
inline operation_status cuda_storage_status(cudaError_t status)
{
    if (status != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(status)});
    return {};
}
struct device_allocation
{
    std::byte* allocations[5]{};
    device_descriptor descriptor;
    cudaStream_t stream{};
    bool ready{};
    ~device_allocation()
    {
        int previous{-1};
        const auto queried = cudaGetDevice(&previous);
        if (descriptor.device < 0 || cudaSetDevice(descriptor.device) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);

        for (auto allocation : allocations) if (allocation) cudaFree(allocation);
        if (stream) cudaStreamDestroy(stream);
        if (queried == cudaSuccess && previous != descriptor.device) cudaSetDevice(previous);
    }
};
template<schema s, layout l> class cuda_draft
{
    using traits = storage_traits<s, l>;
    std::unique_ptr<device_allocation> state_;
public:
    cuda_draft() = default;
    cuda_draft(const cuda_draft&) = delete;
    cuda_draft& operator=(const cuda_draft&) = delete;
    operation_status initialize(std::size_t count);
    operation_result<device_map_view<s>> view();
    operation_result<device_map_view<s, true>> view() const;
    template<layout host_layout> operation_status upload(const draft<s, host_layout>& source);
    template<layout host_layout> operation_status download(draft<s, host_layout>& destination) const;
    cudaStream_t stream() const noexcept { return state_ ? state_->stream : nullptr; }
    std::size_t size() const noexcept { return state_ ? state_->descriptor.size_tiles : 0; }
};
}
