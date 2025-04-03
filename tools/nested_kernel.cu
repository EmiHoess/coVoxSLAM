#include "nested_kernel.hpp"
#include <cuda_runtime.h>
#include <new>

namespace covoxslam::metrics
{
template<storage::schema s> __global__ void nested_query(storage::nested_descriptor descriptor, bool lane_schedule, float* output)
{
    const std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= descriptor.capacity_tiles * 512) return;
    auto tile = index / 512, voxel = index % 512;
    if (lane_schedule) {
        const auto group = index / (descriptor.lanes * 512), within = index % (descriptor.lanes * 512);
        tile = group * descriptor.lanes + within % descriptor.lanes;
        voxel = within / descriptor.lanes;
    }
    if (tile >= descriptor.size_tiles) return;
    const storage::nested_device_accessor<s, true> access{descriptor};
    float value{};
    if (access.template scalar<storage::field_count<s> - 1>(tile, voxel) & 1U) {
        value = access.template scalar<0>(tile, voxel);
        if constexpr (s == storage::schema::tsdf) value *= access.template scalar<1>(tile, voxel);
    }
    new (output + tile * 512 + voxel) float(value);
}
int launch_nested_query(storage::nested_descriptor descriptor, bool lane_schedule, float* output, void* stream)
{
    if (!descriptor.size_tiles) return int(cudaSuccess);
    const auto blocks = descriptor.capacity_tiles * 2;
    if (!output || !descriptor.lanes || blocks > 2147483647U) return int(cudaErrorInvalidValue);
    if (descriptor.record_schema == storage::schema::tsdf)
        nested_query<storage::schema::tsdf><<<unsigned(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(descriptor, lane_schedule, output);
    else if (descriptor.record_schema == storage::schema::esdf)
        nested_query<storage::schema::esdf><<<unsigned(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(descriptor, lane_schedule, output);
    else return int(cudaErrorInvalidValue);
    return int(cudaGetLastError());
}
}
