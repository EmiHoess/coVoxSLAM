#include "ray_resident/hash.cuh"

namespace covoxslam::detail
{
namespace
{
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

int cuda_begin_resident_batch(cuda_resident_view view, void* stream) noexcept
{
    if (!valid_resident_view(view)) return static_cast<int>(cudaErrorInvalidValue);
    const auto voxels = static_cast<std::size_t>(view.tile_capacity) * 512;
    begin_resident_batch<<<static_cast<unsigned>((voxels + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(view);
    return static_cast<int>(cudaGetLastError());
}
}
