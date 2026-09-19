#include "scan.cuh"

namespace covoxslam::detail
{
__global__ void scan_blocks(const std::uint32_t* input, std::uint64_t* output,
                            std::uint64_t* sums, std::size_t count)
{
    __shared__ std::uint64_t values[scan_block_threads];
    const auto lane = threadIdx.x;
     const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + lane;
    const std::uint64_t value = index < count ? input[index] : 0;
    values[lane] = value;
    __syncthreads();
    for (unsigned offset = 1; offset < blockDim.x; offset *= 2) {
        const auto addend = lane >= offset ? values[lane - offset] : 0;
        __syncthreads();
        values[lane] += addend;
        __syncthreads();
    }
    if (index < count) output[index] = values[lane] - value;
    if (lane == blockDim.x - 1) sums[blockIdx.x] = values[lane];
}

__global__ void scan_block_sums(std::uint64_t* sums, unsigned count)
{
    std::uint64_t total{};
     for (unsigned index = 0; index < count; ++index)
    {
        const auto value = sums[index];
        sums[index] = total;
        total += value;
    }
}

__global__ void add_block_offsets(std::uint64_t* offsets, const std::uint64_t* sums,
                                 std::size_t count) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) offsets[index] += sums[blockIdx.x];
}

int cuda_scan_counts(const std::uint32_t* counts, std::uint64_t* offsets,
                     std::uint64_t* block_sums, std::size_t count, void* stream) noexcept
{
    if (!count) return static_cast<int>(cudaSuccess);
    const auto blocks = static_cast<unsigned>(scan_block_count(count));
     const auto cuda_stream = static_cast<cudaStream_t>(stream);
    scan_blocks<<<blocks, scan_block_threads, 0, cuda_stream>>>(counts, offsets, block_sums, count);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    scan_block_sums<<<1, 1, 0, cuda_stream>>>(block_sums, blocks);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    add_block_offsets<<<blocks, scan_block_threads, 0, cuda_stream>>>(offsets, block_sums, count);
    return static_cast<int>(cudaGetLastError());
}
}
