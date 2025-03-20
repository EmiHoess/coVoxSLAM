#include "layout_kernel.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <new>

namespace covoxslam::metrics
{
using namespace storage;
template<schema s, bool copy> __global__ void query_kernel(device_descriptor descriptor, void* output)
{
    const device_map_view<s, true> view{device_accessor<s, true>{descriptor}};

    for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        i < view.size() * 512; i += std::size_t(gridDim.x) * blockDim.x)
    {
        const auto v = view[i / 512].voxels[i % 512];
        if constexpr (copy) ::new (static_cast<void*>(static_cast<value<s>*>(output) + i)) value<s>(gather_record(v));
        else {
            float result{};
            if constexpr (s == schema::tsdf) result = (v.flags & 1U) ? v.distance * v.weight : 0;
            else result = (v.flags & 1U) ? v.signed_distance_m : 0;
            ::new (static_cast<void*>(static_cast<float*>(output) + i)) float(result);
        }
    }
}
template<schema s> void launch(device_descriptor descriptor, bool copy, void* output, cudaStream_t stream)
{
    const auto blocks = unsigned(std::min<std::size_t>((descriptor.size_tiles * 512 + 255) / 256, 65535));
    if (copy) query_kernel<s, true><<<blocks, 256, 0, stream>>>(descriptor, output);
    else query_kernel<s, false><<<blocks, 256, 0, stream>>>(descriptor, output);
}
int launch_layout_query(device_descriptor descriptor, bool copy, void* output, void* stream) noexcept
{
    if (descriptor.size_tiles == 0) return int(cudaSuccess);
    if (!output || descriptor.size_tiles > descriptor.capacity_tiles) return int(cudaErrorInvalidValue);
    switch (descriptor.record_schema)
    {
    case schema::tsdf: launch<schema::tsdf>(descriptor, copy, output, static_cast<cudaStream_t>(stream)); break;
    case schema::esdf: launch<schema::esdf>(descriptor, copy, output, static_cast<cudaStream_t>(stream)); break;
    default: return int(cudaErrorInvalidValue);
    }
    return int(cudaGetLastError());
}
}
