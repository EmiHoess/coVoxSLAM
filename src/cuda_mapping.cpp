#include <covoxslam/cuda.hpp>
#include "sparse_mapping.h"

#include <cuda_runtime_api.h>
#include <array>
#include <limits>
#include <stdexcept>

#include <type_traits>

namespace covoxslam
{
namespace
{
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::is_standard_layout_v<detail::cuda_device_ray>);
static_assert(std::is_trivially_copyable_v<detail::cuda_device_ray>);
static_assert(sizeof(detail::cuda_device_ray) == 56);
static_assert(alignof(detail::cuda_device_ray) == 8);
static_assert(offsetof(detail::cuda_device_ray, stamp_ns) == 32);

std::size_t checked_cuda_storage_bytes(std::size_t capacity)
{
    if (!capacity || capacity > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("CUDA scan capacity is unsupported");
     const auto blocks = 1 + (capacity - 1) / 256;
    if (blocks > std::numeric_limits<std::size_t>::max() / 8 ||
        capacity > (std::numeric_limits<std::size_t>::max() - blocks * 8) / 12)
        throw std::length_error("CUDA scan byte count overflows addressable size");
    return capacity * 12 + blocks * 8;
}
}
}

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
    const auto info = query_cuda_device(device);
    if (checked_cuda_storage_bytes(capacity) > info.total_memory_bytes)
        throw std::length_error("CUDA scan storage exceeds total device memory");
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

#include <vector>

namespace covoxslam
{
static_assert(sizeof(detail::cuda_ray_contribution) == 28);
static_assert(sizeof(detail::cuda_ray_summary) == 16);
struct cuda_ray_context::implementation
{
    int device{};
    std::size_t ray_capacity{};
    std::uint64_t contribution_capacity{};
    std::uint64_t contribution_count{};
     detail::cuda_ray_parameters parameters;
    cudaStream_t stream{};
    std::array<void*, 6> buffers{};
    std::array<cudaEvent_t, 6> events{};
    std::vector<detail::cuda_device_ray> rays_host;
    template<class value_type> value_type* buffer(std::size_t index) const noexcept
    {
        return static_cast<value_type*>(buffers[index]);
    }
    ~implementation()
    {
        if (!stream) return;
        cudaSetDevice(device);
        cudaStreamSynchronize(stream);
         for (auto buffer : buffers) if (buffer) cudaFree(buffer);
        for (auto event : events) if (event) cudaEventDestroy(event);
        cudaStreamDestroy(stream);
    }
};

cuda_ray_context::cuda_ray_context(int device, std::size_t ray_capacity,
    std::uint64_t contribution_capacity, cuda_ray_config config)
    : state(std::make_unique<implementation>())
{
    const auto info = query_cuda_device(device);
    state->parameters = {config.voxel_size, config.truncation_distance,
        config.minimum_ray_length_m, config.maximum_ray_length_m, config.voxel_carving, config.allow_clear};
    check_cuda(static_cast<cudaError_t>(detail::cuda_count_ray_contributions(
        nullptr, 0, state->parameters, nullptr, nullptr, nullptr)));
    if (!ray_capacity || ray_capacity > std::numeric_limits<std::uint32_t>::max() ||
        !contribution_capacity || contribution_capacity >
            std::numeric_limits<std::size_t>::max() / sizeof(detail::cuda_ray_contribution))
        throw std::length_error("CUDA contribution capacities are unsupported");
    const std::array<std::size_t, 6> bytes{ray_capacity * sizeof(detail::cuda_device_ray),
        ray_capacity * sizeof(std::uint32_t), ray_capacity * sizeof(std::uint64_t),
        (1 + (ray_capacity - 1) / 256) * sizeof(std::uint64_t),
        static_cast<std::size_t>(contribution_capacity) * sizeof(detail::cuda_ray_contribution),
        sizeof(detail::cuda_ray_summary)};
    std::size_t total{};
    for (const auto size : bytes) {
        if (size > std::numeric_limits<std::size_t>::max() - total)
            throw std::length_error("CUDA contribution storage overflows address space");
        total += size;
    }
    if (total > info.total_memory_bytes)
        throw std::length_error("CUDA contribution storage exceeds device memory");
     state->device = device;
    state->ray_capacity = ray_capacity;
    state->contribution_capacity = contribution_capacity;
    check_cuda(cudaSetDevice(device));
    check_cuda(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
    for (std::size_t index = 0; index < bytes.size(); ++index)
        check_cuda(cudaMalloc(&state->buffers[index], bytes[index]));
    for (auto& event : state->events) check_cuda(cudaEventCreate(&event));
}

cuda_ray_context::~cuda_ray_context() = default;

std::size_t cuda_ray_context::ray_capacity() const noexcept
{
    return state->ray_capacity;
}

std::uint64_t cuda_ray_context::contribution_capacity() const noexcept {
    return state->contribution_capacity;
}

detail::cuda_ray_device_view cuda_ray_context::device_view() const noexcept
{
    return {state->buffer<detail::cuda_ray_contribution>(4), state->contribution_count,
        state->contribution_capacity, state->buffer<std::uint32_t>(1), state->buffer<std::uint64_t>(2),
        state->buffer<std::uint64_t>(3), state->stream, state->device, state->ray_capacity};
}

void cuda_ray_context::copy_contributions(std::span<detail::cuda_ray_contribution> output)
{
    if (output.size() < state->contribution_count)
        throw std::length_error("CUDA contribution download buffer is too small");
    if (!state->contribution_count) return;
     check_cuda(cudaSetDevice(state->device));
    check_cuda(cudaMemcpyAsync(output.data(), state->buffers[4],
        static_cast<std::size_t>(state->contribution_count) * sizeof(detail::cuda_ray_contribution),
        cudaMemcpyDeviceToHost, state->stream));
    check_cuda(cudaStreamSynchronize(state->stream));
}
}
