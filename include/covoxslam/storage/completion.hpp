#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/cuda_transfers.hpp>
#include <memory>

namespace covoxslam::storage
{
struct completion_state
{
    std::shared_ptr<void> retained;
    cudaEvent_t event{};
    cudaStream_t stream{};
    int device{-1};
    bool recorded{};
    ~completion_state()
    {
        int previous{-1};
        const auto queried = cudaGetDevice(&previous);
        if (device < 0) { if (stream) cudaStreamSynchronize(stream); return; }
        if (cudaSetDevice(device) != cudaSuccess) return;
        if (recorded) cudaEventSynchronize(event);
        else if (stream) cudaStreamSynchronize(stream);
        if (event) cudaEventDestroy(event);
        if (queried == cudaSuccess && previous != device) cudaSetDevice(previous);
    }
};
class completion_ticket
{
    std::shared_ptr<completion_state> state_;
    explicit completion_ticket(std::shared_ptr<completion_state> state) : state_(std::move(state)) {}
public:
    completion_ticket() = default;
    completion_ticket(const completion_ticket&) = delete;
    completion_ticket& operator=(const completion_ticket&) = delete;
    completion_ticket(completion_ticket&&) noexcept = default;
    completion_ticket& operator=(completion_ticket&&) noexcept = default;
    static operation_result<completion_ticket> record(cudaStream_t stream, std::shared_ptr<void> retained)
    {
        auto state = std::make_shared<completion_state>();
        state->stream = stream;
        state->retained = std::move(retained);
        auto error = cudaGetDevice(&state->device);
        if (error == cudaSuccess) error = cudaEventCreateWithFlags(&state->event, cudaEventDisableTiming);
        if (error == cudaSuccess) error = cudaEventRecord(state->event, stream);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        state->recorded = true;
        return completion_ticket(std::move(state));
    }
    operation_result<bool> query() const
    {
        if (!state_) return std::unexpected(operation_error{"completion ticket is empty"});
        int current{-1};
        auto error = cudaGetDevice(&current);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        if (current != state_->device) return std::unexpected(operation_error{"completion requires the allocation device"});
        error = cudaEventQuery(state_->event);
        if (error == cudaErrorNotReady) return false;
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        return true;
    }
    operation_status wait() const
    {
        auto available = query();
        if (!available) return std::unexpected(available.error());
        const auto error = *available ? cudaSuccess : cudaEventSynchronize(state_->event);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        return {};
    }
    operation_status depend_on(cudaStream_t stream) const
    {
        auto available = query();
        if (!available) return std::unexpected(available.error());
        const auto error = cudaStreamWaitEvent(stream, state_->event, 0);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        return {};
    }
    operation_status cancel()
    {
        auto checked = wait();
        if (!checked) return checked;
        state_.reset();
        return {};
    }
};
struct stream_drain
{
    cudaStream_t stream{};
    int device{-1};
    bool armed{true};
    ~stream_drain()
    {
        if (!armed) return;
        int previous{-1};
        const auto queried = cudaGetDevice(&previous);
        if (device < 0 || cudaSetDevice(device) != cudaSuccess) return;
        cudaStreamSynchronize(stream);
        if (queried == cudaSuccess && previous != device) cudaSetDevice(previous);
    }
};
class pinned_payload
{
    void* data_{};
    std::size_t bytes_{};
    bool initialized_{};
public:
    pinned_payload() = default;
    pinned_payload(const pinned_payload&) = delete;
    pinned_payload& operator=(const pinned_payload&) = delete;
    ~pinned_payload() { if (data_) cudaFreeHost(data_); }
    operation_status initialize(std::size_t bytes, std::size_t budget)
    {
        if (initialized_) return std::unexpected(operation_error{"pinned payload is already initialized"});
        if (bytes > budget || bytes > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
            return std::unexpected(operation_error{"pinned payload exceeds budget"});
        if (!bytes) { initialized_ = true; return {}; }
        const auto error = cudaHostAlloc(&data_, bytes, cudaHostAllocDefault);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        bytes_ = bytes;
        initialized_ = true;
        return {};
    }
    std::byte* data() const noexcept { return static_cast<std::byte*>(data_); }
    std::size_t size() const noexcept { return bytes_; }
};
}
