#pragma once
#include "../common/mapping_types.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
__global__ void key_sort_flags(const std::uint64_t* keys, std::uint32_t* flags, unsigned count, unsigned bit)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i < count) flags[i] = 1U - unsigned((keys[i] >> bit) & 1U);
}
__global__ void key_sort_scatter(const std::uint64_t* input, std::uint64_t* output,
    const std::uint32_t* flags, const std::uint64_t* offsets, unsigned count)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= count) return;
    const auto zeros = offsets[count - 1] + flags[count - 1];
    output[flags[i] ? offsets[i] : zeros + i - offsets[i]] = input[i];
}
inline cudaError_t scan_key_flags(const std::uint32_t* input, std::uint64_t* output,
    std::uint64_t* sums, unsigned count, cudaStream_t stream)
{
    return static_cast<cudaError_t>(cuda_scan_counts(input, output, sums, count, stream));
}
inline cudaError_t sort_tile_keys(std::uint64_t* keys, std::uint64_t* scratch,
    std::uint32_t* flags, std::uint64_t* offsets, std::uint64_t* sums, unsigned count, cudaStream_t stream)
{
    if (!count || count > 1048576 || !keys || !scratch || !flags || !offsets || !sums)
        return cudaErrorInvalidValue;
    auto input = keys, output = scratch;

    for (unsigned bit = 0; bit < 64; ++bit) {
        key_sort_flags<<<(count + 255) / 256, 256, 0, stream>>>(input, flags, count, bit);
        auto error = cudaGetLastError();
        if (!error) error = scan_key_flags(flags, offsets, sums, count, stream);

        if (error) return error;
        key_sort_scatter<<<(count + 255) / 256, 256, 0, stream>>>(input, output, flags, offsets, count);
        error = cudaGetLastError();

        if (error) return error;
        auto temporary = input; input = output; output = temporary;
    }
    return cudaSuccess;
}
}
