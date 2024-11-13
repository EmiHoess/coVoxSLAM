#include "sparse_mapping.h"
#include "blocks/initialize.cuh"
#include "blocks/proposal.cuh"
#include "blocks/validate.cuh"
#include <cmath>
#include <initializer_list>

namespace covoxslam::detail
{
bool valid_voxblox_parameters(cuda_blocks_parameters config)
{
    return std::isfinite(config.voxel_size) && config.voxel_size > 0 &&
        std::isfinite(config.seed_band_m) && config.seed_band_m >= 0 &&
        std::isfinite(config.maximum_distance_m) && config.maximum_distance_m > 0 &&
        config.seed_band_m <= config.maximum_distance_m &&
        double(config.maximum_distance_m) / config.voxel_size < 10000000 &&
        std::isfinite(config.minimum_weight) && config.minimum_weight >= 0 &&
        std::isfinite(config.minimum_difference_m) && config.minimum_difference_m >= 0 &&
        config.bucket_count >= 2 && config.bucket_count <= 65536;
}

bool valid_voxblox_workspace(cuda_blocks_workspace work)
{
    return work.values && work.next && work.queued && work.heads && work.tails && work.order && work.queue;
}

int cuda_initialize_voxblox(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_blocks_parameters config, cuda_blocks_workspace work, void* stream) noexcept
{
    if (!valid_voxblox_parameters(config) || tile_count > UINT32_MAX / 512 ||
        (tile_count && (!keys || !tsdf || !valid_voxblox_workspace(work))))
        return static_cast<int>(cudaErrorInvalidValue);
    if (!tile_count) return static_cast<int>(cudaSuccess);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    const std::size_t count = tile_count * 512;
    for (const auto pointer : {work.heads, work.tails}) {
        const auto error = cudaMemsetAsync(pointer, 0xFF, config.bucket_count * sizeof(std::uint32_t), cuda_stream);
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    auto error = cudaMemsetAsync(work.queued, 0, count * sizeof(std::uint32_t), cuda_stream);
    if (error != cudaSuccess) return static_cast<int>(error);
     error = cudaMemsetAsync(work.queue, 0, sizeof(*work.queue), cuda_stream);
    if (error != cudaSuccess) return static_cast<int>(error);
    initialize_voxblox<<<static_cast<unsigned>((count + 255) / 256), 256, 0, cuda_stream>>>(
        keys, tsdf, tile_count, config, work);
    return static_cast<int>(cudaGetLastError());
}

int cuda_enqueue_voxblox_seeds(std::size_t count, cuda_blocks_parameters config,
    cuda_blocks_workspace work, void* stream) noexcept
{
    if (!valid_voxblox_parameters(config) || count > UINT32_MAX || (count && !valid_voxblox_workspace(work)))
        return static_cast<int>(cudaErrorInvalidValue);
    for (std::size_t begin = 0; begin < count; begin += 262144) {
        const auto end = count - begin < 262144 ? count : begin + 262144;
        enqueue_voxblox_seeds<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(count, begin, end, config, work);
         const auto error = cudaGetLastError();
        if (error != cudaSuccess) return static_cast<int>(error);
    }
    return static_cast<int>(cudaSuccess);
}
}

#include "blocks/ordered.cuh"

namespace covoxslam::detail
{
int cuda_propagate_voxblox_ordered(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_blocks_parameters config, cuda_blocks_workspace work, std::uint32_t maximum_pops, void* stream) noexcept
{
    if (!valid_voxblox_parameters(config) || !maximum_pops || tile_count > UINT32_MAX / 512 ||
        (tile_count && (!keys || !valid_voxblox_workspace(work)))) return static_cast<int>(cudaErrorInvalidValue);
    if (!tile_count) return static_cast<int>(cudaSuccess);
    propagate_voxblox_ordered<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(keys, tile_count, config, work, maximum_pops);
    const auto completed_error = cudaGetLastError();
    if (completed_error != cudaSuccess) return static_cast<int>(completed_error);
    validate_voxblox<<<static_cast<unsigned>((tile_count * 512 + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        keys, nullptr, tile_count, config, work, false);
     return static_cast<int>(cudaGetLastError());
}
}

#include "blocks/frontier.cuh"

namespace covoxslam::detail
{
int cuda_propagate_voxblox_frontier(const cuda_tile_coordinate* keys, std::size_t tile_count,
    cuda_blocks_parameters config, cuda_blocks_workspace work, std::uint32_t capacity, void* stream) noexcept
{
    if (!valid_voxblox_parameters(config) || !capacity || capacity > UINT32_MAX / 26 || tile_count > UINT32_MAX / 512 ||
        (tile_count && (!keys || !valid_voxblox_workspace(work) || !work.sources || !work.winners || !work.proposals)))
        return static_cast<int>(cudaErrorInvalidValue);
    if (!tile_count) return static_cast<int>(cudaSuccess);
    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    dequeue_voxblox_frontier<<<1, 1, 0, cuda_stream>>>(config, work, capacity);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    const auto blocks = static_cast<unsigned>((std::uint64_t(capacity) * 26 + 255) / 256);
    propose_voxblox_frontier<<<blocks, 256, 0, cuda_stream>>>(keys, tile_count, config, work);
     error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    commit_voxblox_frontier<<<1, 1, 0, cuda_stream>>>(config, work);
    const auto completed_error = cudaGetLastError();
    if (completed_error != cudaSuccess) return static_cast<int>(completed_error);
    validate_voxblox<<<static_cast<unsigned>((tile_count * 512 + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        keys, nullptr, tile_count, config, work, false);
    return static_cast<int>(cudaGetLastError());
}
}

namespace covoxslam::detail
{
int cuda_validate_voxblox(const cuda_tile_coordinate* keys, const cuda_tsdf_tile* tsdf,
    std::size_t tile_count, cuda_blocks_parameters config, cuda_blocks_workspace work, void* stream) noexcept
{
    if (!valid_voxblox_parameters(config) || tile_count > UINT32_MAX / 512 ||
        (tile_count && (!keys || !tsdf || !valid_voxblox_workspace(work)))) return static_cast<int>(cudaErrorInvalidValue);
    if (!tile_count) return static_cast<int>(cudaSuccess);
    validate_voxblox<<<static_cast<unsigned>((tile_count * 512 + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        keys, tsdf, tile_count, config, work, true);
    return static_cast<int>(cudaGetLastError());
}
}
