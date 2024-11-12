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

#include "primitives/unique.cuh"
#include "ray_fusion/serial/fuse.cuh"
#include <cmath>

namespace covoxslam::detail
{
int cuda_build_voxel_segments(const cuda_ray_contribution* values, std::size_t count,
    std::uint32_t* flags, std::uint64_t* offsets, std::uint64_t* block_sums,
    cuda_voxel_segment* segments, std::uint32_t capacity, std::uint32_t* segment_count,
    std::uint32_t* status, void* stream) noexcept
{
    if (count > UINT32_MAX || !segment_count || !status || (count &&
        (!values || !flags || !offsets || !block_sums || (capacity && !segments))))
        return static_cast<int>(cudaErrorInvalidValue);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    if (!count) return static_cast<int>(cudaMemsetAsync(segment_count, 0, sizeof(*segment_count), cuda_stream));
     const auto blocks = static_cast<unsigned>(1 + (count - 1) / 256);
    mark_voxel_heads<<<blocks, 256, 0, cuda_stream>>>(values, flags, count);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    const auto scan_error = cuda_scan_counts(flags, offsets, block_sums, count, stream);
    if (scan_error) return scan_error;
    write_voxel_heads<<<blocks, 256, 0, cuda_stream>>>(values, flags, offsets, count, segments, capacity);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    finish_voxel_segments<<<blocks, 256, 0, cuda_stream>>>(
        flags, offsets, count, segments, capacity, segment_count, status);
    return static_cast<int>(cudaGetLastError());
}

int cuda_fuse_voxel_segments(const cuda_ray_contribution* values, std::size_t count,
    const cuda_voxel_segment* segments, std::uint32_t segment_count,
    const cuda_tile_coordinate* keys, cuda_tsdf_tile* tiles, std::uint32_t tile_count,
    cuda_fusion_parameters config, std::uint32_t* status, void* stream) noexcept
{
    if (!std::isfinite(config.truncation_distance) || config.truncation_distance <= 0 ||
        !std::isfinite(config.maximum_weight) || config.maximum_weight <= 0 ||
        !std::isfinite(config.extraction_distance) || config.extraction_distance < 0 ||
        config.extraction_distance > config.truncation_distance || count > UINT32_MAX ||
        (segment_count && (!values || !segments || !keys || !tiles || !status || !tile_count)))
        return static_cast<int>(cudaErrorInvalidValue);
    if (!segment_count) return static_cast<int>(cudaSuccess);
     const auto blocks = static_cast<unsigned>(1 + (segment_count - 1) / 256);
    validate_fusion_segments<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        values, count, segments, segment_count, keys, tiles, tile_count, config, status);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    fuse_voxel_segments<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        values, count, segments, segment_count, keys, tiles, tile_count, config, status);
    return static_cast<int>(cudaGetLastError());
}
}

#include "ray_fusion/weighted_sum/fuse.cuh"

#include "ray_fusion/dispatch.cuh"
