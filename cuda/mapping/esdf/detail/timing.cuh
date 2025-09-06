#pragma once
#include <chrono>
#include <cuda_runtime.h>
#include <covoxslam/mapping/esdf/band_esdf.hpp>

namespace covoxslam::store
{
struct band_event_timer
{
    cudaEvent_t begin{}, end{};
    ~band_event_timer()
    {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
    cudaError_t start(cudaStream_t stream)
    {
        auto error = cudaSuccess;
        if (!begin) error = cudaEventCreate(&begin);
        if (!error && !end) error = cudaEventCreate(&end);
        return error ? error : cudaEventRecord(begin, stream);
    }
    cudaError_t read(unsigned* pointer, unsigned& value, cudaStream_t stream, band_esdf_metrics& metrics)
    {
        auto error = cudaEventRecord(end, stream);
        if (error) return error;
        const auto copied = cudaMemcpyAsync(&value, pointer, sizeof(value), cudaMemcpyDeviceToHost, stream);
        const auto started = std::chrono::steady_clock::now();
        const auto waited = cudaStreamSynchronize(stream);
        metrics.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        if (copied || waited) return copied ? copied : waited;
        float elapsed{};
        error = cudaEventElapsedTime(&elapsed, begin, end);
        if (!error) metrics.gpu_ms += elapsed;
        return error;
    }
};
}
