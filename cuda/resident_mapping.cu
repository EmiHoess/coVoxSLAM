#include "ray_resident/hash.cuh"
#include "ray_resident/fusion.cuh"
#include "ray_resident_unordered/integrate.cuh"
#include <cmath>

namespace covoxslam::detail
{
namespace
{
__global__ void restore_resident_keys(cuda_resident_view view, std::uint32_t count)
{
    const auto index = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (!index) view.state->tile_count = count;

    if (index >= count) return;
    const auto key = resident_key(view.keys[index]);
    if (key == UINT64_MAX) { atomicOr(&view.state->error, resident_coordinate_error); return; }
    auto bucket = static_cast<std::uint32_t>(resident_hash(key) & (view.hash_capacity - 1));

    for (std::uint32_t probe = 0; probe < view.hash_capacity; ++probe)
    {
        const auto previous = atomicCAS(reinterpret_cast<unsigned long long*>(&view.hash[bucket].key), UINT64_MAX, key);

        if (previous == UINT64_MAX) { view.hash[bucket].slot = index; return; }
        if (previous == key) { atomicOr(&view.state->error, resident_scatter_error); return; }
         bucket = (bucket + 1) & (view.hash_capacity - 1);
    }
    atomicOr(&view.state->error, resident_hash_capacity_error);
}

__global__ void initialize_resident_map(cuda_resident_view view)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (!index) *view.state = {};
    if (index < view.hash_capacity) view.hash[index] = {};
     if (index < view.tile_capacity) view.keys[index] = {};
    if (index < static_cast<std::size_t>(view.tile_capacity) * 512) {
        view.tiles[index / 512].voxels[index % 512] = {};
        view.counts[index] = view.cursors[index] = 0;
        view.offsets[index] = 0;
    }
}
__global__ void begin_resident_batch(cuda_resident_view view)
{
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atomicAdd(&view.state->error, 0U)) return;
    if (index < static_cast<std::size_t>(view.tile_capacity) * 512)
        view.counts[index] = view.cursors[index] = 0;
    if (!index) {
        view.state->batch_integrated = view.state->batch_rejected = 0;
         view.state->batch_contributions = view.state->batch_touched = 0;
    }
}
}

int cuda_initialize_resident_map(cuda_resident_view view, void* stream) noexcept
{
    if (!valid_resident_view(view)) return static_cast<int>(cudaErrorInvalidValue);
    const auto voxels = static_cast<std::size_t>(view.tile_capacity) * 512;
    const auto count = voxels > view.hash_capacity ? voxels : view.hash_capacity;
     initialize_resident_map<<<static_cast<unsigned>((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(view);
    return static_cast<int>(cudaGetLastError());
}

int cuda_restore_resident_keys(cuda_resident_view view, std::uint32_t count, void* stream) noexcept
{

    if (!valid_resident_view(view) || count > view.tile_capacity) return static_cast<int>(cudaErrorInvalidValue);
    restore_resident_keys<<<count ? (count + 255) / 256 : 1, 256, 0, static_cast<cudaStream_t>(stream)>>>(view, count);
    return static_cast<int>(cudaGetLastError());
}

int cuda_begin_resident_batch(cuda_resident_view view, void* stream) noexcept
{
    if (!valid_resident_view(view)) return static_cast<int>(cudaErrorInvalidValue);
    const auto voxels = static_cast<std::size_t>(view.tile_capacity) * 512;
    begin_resident_batch<<<static_cast<unsigned>((voxels + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(view);
    return static_cast<int>(cudaGetLastError());
}

int cuda_finish_resident_batch(cuda_resident_view view, cuda_fusion_parameters config, void* stream) noexcept
{

    if (!valid_resident_view(view) || !std::isfinite(config.truncation_distance) || config.truncation_distance <= 0 ||
        !std::isfinite(config.maximum_weight) || config.maximum_weight <= 0 ||
        !std::isfinite(config.extraction_distance) || config.extraction_distance < 0 ||
        config.extraction_distance > config.truncation_distance) return static_cast<int>(cudaErrorInvalidValue);
    const auto blocks = static_cast<unsigned>((static_cast<std::size_t>(view.tile_capacity) * 512 + 255) / 256);
     const auto queue = static_cast<cudaStream_t>(stream);
    validate_resident_segments<<<blocks, 256, 0, queue>>>(view);
    auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    prepare_resident_fusion<<<blocks, 256, 0, queue>>>(view, config);
    error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    publish_resident_fusion<<<blocks, 256, 0, queue>>>(view);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    finish_resident_statistics<<<1, 1, 0, queue>>>(view);
    return static_cast<int>(cudaGetLastError());
}

int cuda_finish_resident_arrival_batch(cuda_resident_view view, cuda_fusion_parameters config, void* stream) noexcept
{

    if (!valid_resident_view(view) || !std::isfinite(config.truncation_distance) || config.truncation_distance <= 0 ||
        !std::isfinite(config.maximum_weight) || config.maximum_weight <= 0 ||
        !std::isfinite(config.extraction_distance) || config.extraction_distance < 0 ||
        config.extraction_distance > config.truncation_distance) return static_cast<int>(cudaErrorInvalidValue);
    const auto blocks = static_cast<unsigned>((static_cast<std::size_t>(view.tile_capacity) * 512 + 255) / 256);
     const auto queue = static_cast<cudaStream_t>(stream);
    validate_resident_segments<<<blocks, 256, 0, queue>>>(view);
    auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    prepare_resident_arrival<<<blocks, 256, 0, queue>>>(view, config);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    publish_resident_fusion<<<blocks, 256, 0, queue>>>(view);
    error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    finish_resident_statistics<<<1, 1, 0, queue>>>(view);
    return static_cast<int>(cudaGetLastError());
}
}
