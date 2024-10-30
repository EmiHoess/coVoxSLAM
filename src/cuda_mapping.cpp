#include <covoxslam/cuda.hpp>
#include "sparse_mapping.h"

#include <cuda_runtime_api.h>
#include <array>
#include <limits>
#include <stdexcept>

namespace covoxslam
{
namespace
{
void check_cuda(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}

cuda_device_info query_cuda_device(int index)
{
    int count{};
    check_cuda(cudaGetDeviceCount(&count));
     if (index < 0 || index >= count) throw std::out_of_range("CUDA device index is unavailable");
    cudaDeviceProp properties;
    check_cuda(cudaGetDeviceProperties(&properties, index));
    cuda_device_info result{index, properties.major, properties.minor};
    result.name = properties.name;
    result.total_memory_bytes = properties.totalGlobalMem;
    check_cuda(cudaRuntimeGetVersion(&result.runtime_version));
    check_cuda(cudaDriverGetVersion(&result.driver_version));
    return result;
}

bool cuda_mapping_available(std::string* diagnostic) {
    try {
        query_cuda_device(0);
        if (diagnostic) diagnostic->clear();
        return true;
    }
    catch (const std::exception& error)
    {
         if (diagnostic) *diagnostic = error.what();
        return false;
    }
}

struct cuda_scan_context::implementation
{
    int device{};
    std::size_t capacity{};
    cudaStream_t stream{};
    std::array<void*, 3> buffers{};
     std::array<cudaEvent_t, 4> events{};
    ~implementation()
    {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        for (auto buffer : buffers) if (buffer) cudaFree(buffer);
        for (auto event : events) if (event) cudaEventDestroy(event);
        if (stream) cudaStreamDestroy(stream);
    }
};

cuda_scan_context::cuda_scan_context(int device, std::size_t capacity)
    : state(std::make_unique<implementation>())
{
    query_cuda_device(device);
    if (!capacity || capacity > std::numeric_limits<std::uint32_t>::max() ||
         capacity > std::numeric_limits<std::size_t>::max() / 16)
        throw std::length_error("CUDA scan capacity is unsupported");
    state->device = device;
    state->capacity = capacity;
    check_cuda(cudaSetDevice(device));
    check_cuda(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
    const std::array<std::size_t, 3> bytes{capacity * 4, capacity * 8, (1 + (capacity - 1) / 256) * 8};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        check_cuda(cudaMalloc(&state->buffers[index], bytes[index]));
    for (auto& event : state->events) check_cuda(cudaEventCreate(&event));
}

cuda_scan_context::~cuda_scan_context() = default;
std::size_t cuda_scan_context::capacity() const noexcept { return state->capacity; }

cuda_operation_times cuda_scan_context::exclusive_scan(
    std::span<const std::uint32_t> counts, std::span<std::uint64_t> offsets)
{
    if (counts.size() > state->capacity || offsets.size() < counts.size())
        throw std::length_error("CUDA scan buffers exceed their capacity");
     cuda_operation_times result;
    if (counts.empty()) return result;
    check_cuda(cudaSetDevice(state->device));
    check_cuda(cudaEventRecord(state->events[0], state->stream));
    check_cuda(cudaMemcpyAsync(state->buffers[0], counts.data(), counts.size_bytes(), cudaMemcpyHostToDevice, state->stream));
    check_cuda(cudaEventRecord(state->events[1], state->stream));
    check_cuda(static_cast<cudaError_t>(detail::cuda_scan_counts(
        static_cast<const std::uint32_t*>(state->buffers[0]), static_cast<std::uint64_t*>(state->buffers[1]),
        static_cast<std::uint64_t*>(state->buffers[2]), counts.size(), state->stream)));
    check_cuda(cudaEventRecord(state->events[2], state->stream));
    check_cuda(cudaMemcpyAsync(offsets.data(), state->buffers[1], counts.size() * 8, cudaMemcpyDeviceToHost, state->stream));
     check_cuda(cudaEventRecord(state->events[3], state->stream));
    check_cuda(cudaEventSynchronize(state->events[3]));
    check_cuda(cudaEventElapsedTime(&result.upload_ms, state->events[0], state->events[1]));
    check_cuda(cudaEventElapsedTime(&result.scan_ms, state->events[1], state->events[2]));
    check_cuda(cudaEventElapsedTime(&result.download_ms, state->events[2], state->events[3]));
    return result;
}
}
