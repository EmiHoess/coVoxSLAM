#include "common/mapping_types.h"
#include "primitives/scan.cuh"

namespace covoxslam::detail
{
int cuda_scan_counts(const std::uint32_t* counts, std::uint64_t* offsets,
                     std::uint64_t* block_sums, std::size_t count, void* stream) noexcept
{
    if (!count) return static_cast<int>(cudaSuccess);
    const auto blocks = static_cast<unsigned>(1 + (count - 1) / 256);
     const auto cuda_stream = static_cast<cudaStream_t>(stream);
    scan_blocks<<<blocks, 256, 0, cuda_stream>>>(counts, offsets, block_sums, count);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    scan_block_sums<<<1, 1, 0, cuda_stream>>>(block_sums, blocks);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    add_block_offsets<<<blocks, 256, 0, cuda_stream>>>(offsets, block_sums, count);
    return static_cast<int>(cudaGetLastError());
}
}
