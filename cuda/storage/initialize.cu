#include <covoxslam/storage/views.hpp>
#include <cuda_runtime.h>
#include <new>
#include <algorithm>

namespace covoxslam::storage
{
template<schema s> __global__ void initialize_fields_kernel(device_descriptor descriptor)
{
    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        i < descriptor.capacity_tiles * voxels_per_tile; i += std::size_t(blockDim.x) * gridDim.x)
    {
        const auto offset = i / voxels_per_tile * descriptor.tile_stride + i % voxels_per_tile * descriptor.voxel_stride;
        ::new (static_cast<void*>(descriptor.fields[0] + offset)) float{};
        if constexpr (s == schema::tsdf) {
            ::new (static_cast<void*>(descriptor.fields[1] + offset)) float{};
            ::new (static_cast<void*>(descriptor.fields[2] + offset)) std::uint32_t{};
            ::new (static_cast<void*>(descriptor.fields[3] + offset)) std::uint32_t{};
        }
        else {
            ::new (static_cast<void*>(descriptor.fields[1] + offset)) std::int32_t{};
            ::new (static_cast<void*>(descriptor.fields[2] + offset)) std::int32_t{};
            ::new (static_cast<void*>(descriptor.fields[3] + offset)) std::int32_t{};
            ::new (static_cast<void*>(descriptor.fields[4] + offset)) std::uint32_t{};
        }
    }
}
int initialize_device_fields(device_descriptor descriptor, void* stream) noexcept
{
    if (descriptor.storage_layout != layout::pool_soa || descriptor.tile_stride != 2048 ||
        descriptor.voxel_stride != 4 || descriptor.size_tiles > descriptor.capacity_tiles ||
        !descriptor.generation || descriptor.record_schema == schema::blocks_esdf)
        return int(cudaErrorInvalidValue);
    if (descriptor.capacity_tiles == 0) return int(cudaSuccess);
    const auto count = descriptor.capacity_tiles * voxels_per_tile;
    const unsigned blocks = unsigned(std::min<std::size_t>((count + 255) / 256, 65535));
    const auto queue = static_cast<cudaStream_t>(stream);
    switch (descriptor.record_schema)
    {
    case schema::tsdf: initialize_fields_kernel<schema::tsdf><<<blocks, 256, 0, queue>>>(descriptor); break;
    case schema::esdf:
    case schema::projective_esdf: initialize_fields_kernel<schema::esdf><<<blocks, 256, 0, queue>>>(descriptor); break;
    default: return int(cudaErrorInvalidValue);
    }
    return int(cudaGetLastError());
}
}
