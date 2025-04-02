#include <covoxslam/storage/nested_device_view.hpp>
#include <cuda_runtime.h>
#include <new>

namespace covoxslam::storage
{
template<schema s> __global__ void construct_nested(nested_descriptor descriptor)
{
    const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= descriptor.capacity_tiles * 512) return;
    const auto tile = index / 512, voxel = index % 512;
    const auto offset = tile / descriptor.lanes * descriptor.group_stride +
        tile % descriptor.lanes * descriptor.tile_stride + voxel * descriptor.voxel_stride;

    new (descriptor.fields[0] + offset) float{};
    if constexpr (s == schema::tsdf) {
        new (descriptor.fields[1] + offset) float{};
        new (descriptor.fields[2] + offset) std::uint32_t{};
        new (descriptor.fields[3] + offset) std::uint32_t{};
    }
    else {
        new (descriptor.fields[1] + offset) std::int32_t{};
        new (descriptor.fields[2] + offset) std::int32_t{};
        new (descriptor.fields[3] + offset) std::int32_t{};
        new (descriptor.fields[4] + offset) std::uint32_t{};
    }
}
int initialize_nested_fields(nested_descriptor descriptor, void* stream)
{
    if (!descriptor.capacity_tiles) return int(cudaSuccess);
    const auto blocks = descriptor.capacity_tiles * 2;
    if (blocks > 2147483647U) return int(cudaErrorInvalidValue);
    const auto launch = [&]<schema s>() {
        construct_nested<s><<<unsigned(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(descriptor);
    };
    switch (descriptor.record_schema) {
    case schema::tsdf: launch.template operator()<schema::tsdf>(); break;
    case schema::esdf: launch.template operator()<schema::esdf>(); break;
    case schema::blocks_esdf: launch.template operator()<schema::blocks_esdf>(); break;
    case schema::projective_esdf: launch.template operator()<schema::projective_esdf>(); break;
    default: return int(cudaErrorInvalidValue);
    }
    return int(cudaGetLastError());
}
}
