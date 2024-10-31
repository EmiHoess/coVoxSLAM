#include "sparse_mapping.h"
#include "primitives/radix_sort.cuh"
#include <utility>

namespace covoxslam::detail
{
int cuda_stable_sort_contributions(cuda_ray_contribution* values, cuda_ray_contribution* scratch,
    std::uint32_t* flags, std::uint64_t* offsets, std::uint64_t* block_sums,
    std::size_t count, void* stream) noexcept
{
    if (count > UINT32_MAX || (count && (!values || !scratch || values == scratch ||
        !flags || !offsets || !block_sums))) return static_cast<int>(cudaErrorInvalidValue);
    if (!count) return static_cast<int>(cudaSuccess);
     const auto blocks = static_cast<unsigned>(1 + (count - 1) / 256);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    for (unsigned bit = 0; bit < 160; ++bit)
    {
        contribution_radix_flags<<<blocks, 256, 0, cuda_stream>>>(values, flags, count, bit);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        const auto scan_error = cuda_scan_counts(flags, offsets, block_sums, count, stream);
        if (scan_error) return scan_error;
        contribution_radix_scatter<<<blocks, 256, 0, cuda_stream>>>(values, scratch, flags, offsets, count);
         error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
        std::swap(values, scratch);
    }
    return static_cast<int>(cudaSuccess);
}
}
