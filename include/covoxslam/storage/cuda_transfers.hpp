#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/transfers.hpp>
#include <cuda_runtime_api.h>
#include <chrono>

namespace covoxslam::storage
{
struct transfer_measurement
{
    std::size_t calls{}, useful_bytes{}, waits{};
    double planning_ms{}, submission_ms{}, completion_ms{};
    bool fallback{};
};
inline operation_result<transfer_plan> plan_cuda_transfer(const transfer_endpoint& source, std::size_t source_first,
    const transfer_endpoint& destination, std::size_t destination_first, std::size_t count, std::uint32_t fields,
    transfer_method method, transfer_measurement& measurement)
{
    const auto start = std::chrono::steady_clock::now();
    int device{-1}, pitch{};
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaDeviceGetAttribute(&pitch, cudaDevAttrMaxPitch, device);
    if (status != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(status)});
    if ((source.device >= 0 && source.device != device) || (destination.device >= 0 && destination.device != device))
        return std::unexpected(operation_error{"transfer requires the allocation device"});
    auto result = plan_transfer(source, source_first, destination, destination_first, count, fields, method, std::size_t(pitch));
    measurement.planning_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (result) { measurement.useful_bytes = result->useful_bytes; measurement.fallback = result->fallback; }
    return result;
}
inline operation_status execute_cuda_transfer(const transfer_plan& plan, cudaMemcpyKind kind, cudaStream_t stream,
    transfer_measurement& measurement)
{
    const auto start = std::chrono::steady_clock::now();
    cudaError_t submitted = cudaSuccess;

    for (const auto& region : plan.regions)
    {
        ++measurement.calls;
        submitted = region.height == 1 ? cudaMemcpyAsync(region.destination, region.source, region.width, kind, stream) :
            cudaMemcpy2DAsync(region.destination, region.destination_pitch, region.source, region.source_pitch,
                region.width, region.height, kind, stream);
        if (submitted != cudaSuccess) break;
    }
    const auto boundary = std::chrono::steady_clock::now();
    measurement.submission_ms = std::chrono::duration<double, std::milli>(boundary - start).count();
    const auto completed = cudaStreamSynchronize(stream);
    ++measurement.waits;
    measurement.completion_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - boundary).count();
    if (submitted != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(submitted)});
    if (completed != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(completed)});
    return {};
}
}
