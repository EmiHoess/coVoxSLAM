#include "ray_resident/hash.cuh"
#include "ray_resident/fusion.cuh"
#include "ray_resident_unordered/integrate.cuh"
#include "ray_resident_reduce/integrate.cuh"
#include "ray_resident_free_space_skip/skip.cuh"
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
        if (view.offsets) view.offsets[index] = 0;
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
    if (!valid_resident_map(view)) return static_cast<int>(cudaErrorInvalidValue);
    const auto voxels = static_cast<std::size_t>(view.tile_capacity) * 512;
    const auto count = voxels > view.hash_capacity ? voxels : view.hash_capacity;
     initialize_resident_map<<<static_cast<unsigned>((count + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(view);
    return static_cast<int>(cudaGetLastError());
}

int cuda_restore_resident_keys(cuda_resident_view view, std::uint32_t count, void* stream) noexcept
{

    if (!valid_resident_map(view) || count > view.tile_capacity) return static_cast<int>(cudaErrorInvalidValue);
    restore_resident_keys<<<count ? (count + 255) / 256 : 1, 256, 0, static_cast<cudaStream_t>(stream)>>>(view, count);
    return static_cast<int>(cudaGetLastError());
}

int cuda_begin_resident_batch(cuda_resident_view view, void* stream) noexcept
{
    if (!valid_resident_map(view)) return static_cast<int>(cudaErrorInvalidValue);
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

int cuda_begin_resident_reduction(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_resident_reduction mode, void* stream) noexcept
{

    if (!valid_resident_reduction_view(view, reduction, mode)) return static_cast<int>(cudaErrorInvalidValue);
    reduction = resident_reduction_storage(reduction, mode);
    const auto queue = static_cast<cudaStream_t>(stream);
    open_resident_reduction<<<1, 1, 0, queue>>>(view, mode);
    auto error = static_cast<int>(cudaGetLastError());
    if (!error) error = cuda_begin_resident_batch(view, stream);
    if (error) return error;
    const auto voxels = static_cast<std::size_t>(view.tile_capacity) * 512;
    const auto count = voxels > reduction.ray_capacity ? voxels : reduction.ray_capacity;
     clear_resident_reduction<<<static_cast<unsigned>((count + 255) / 256), 256, 0, queue>>>(view, reduction);
    return static_cast<int>(cudaGetLastError());
}

int cuda_finish_resident_reduction(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_resident_reduction mode, void* stream, cuda_fusion_parameters config, cuda_free_space_view skip) noexcept
{
    if (!valid_resident_reduction_view(view, reduction, mode) ||
        (skip.flags && (mode != cuda_resident_reduction::direct_atomic || !std::isfinite(skip.minimum_weight) || skip.minimum_weight <= 0)) ||
        !std::isfinite(config.truncation_distance) || config.truncation_distance <= 0 ||
        !std::isfinite(config.maximum_weight) || config.maximum_weight <= 0 ||
        !std::isfinite(config.extraction_distance) || config.extraction_distance < 0 ||
        config.extraction_distance > config.truncation_distance) return static_cast<int>(cudaErrorInvalidValue);
    const auto voxels = static_cast<std::size_t>(view.tile_capacity) * 512;
    const auto blocks = static_cast<unsigned>((voxels + 255) / 256);
    const auto queue = static_cast<cudaStream_t>(stream);
    require_resident_reduction<<<1, 1, 0, queue>>>(view, mode, skip.flags != nullptr);
    auto error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);

    if (mode == cuda_resident_reduction::index) {
        reduce_resident_indices<<<blocks, 256, 0, queue>>>(view, reduction, config);
        error = cudaGetLastError();
    }
    if (error != cudaSuccess) return static_cast<int>(error);
    validate_resident_reduction_counts<<<blocks, 256, 0, queue>>>(view, mode == cuda_resident_reduction::segment);
    error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    check_resident_reduction_total<<<1, 1, 0, queue>>>(view, mode != cuda_resident_reduction::direct_atomic);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    if (mode == cuda_resident_reduction::segment)
        reduce_resident_segments<<<static_cast<unsigned>((voxels + 7) / 8), 256, 0, queue>>>(view, reduction, config);
    else prepare_resident_sums<<<blocks, 256, 0, queue>>>(view, reduction, config);
    error = cudaGetLastError();

    if (error != cudaSuccess) return static_cast<int>(error);
    if (skip.flags) publish_resident_free_space<<<blocks, 256, 0, queue>>>(view, skip, config.truncation_distance);
    else publish_resident_fusion<<<blocks, 256, 0, queue>>>(view);
    error = cudaGetLastError();
    if (error != cudaSuccess) return static_cast<int>(error);
    finish_resident_reduction_statistics<<<1, 1, 0, queue>>>(view);
    return static_cast<int>(cudaGetLastError());
}
}
