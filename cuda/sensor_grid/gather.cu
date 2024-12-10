#include "native.h"
#include <covoxslam/sensor_grid_gather.h>
#include <cuda_runtime.h>

namespace covoxslam::sensor_grid
{
namespace
{
__global__ void gather_cells(lookup_view table, const float* ranges, voxel* output, gather_policy policy)
{
    const auto index = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= table.active_count) return;
    const auto cell = table.active_voxels[index];
     output[cell] = gather_row(table, ranges, cell, policy);
}
}
int cuda_sensor_gather(lookup_view table, const float* ranges,
    voxel* output, gather_policy policy, void* stream)
{
    if (!table.offsets || !table.intersections || !table.active_voxels || !ranges || !output ||
        !table.ray_count || !table.active_count || table.active_count > table.voxel_count || !table.entry_count)
        return int(cudaErrorInvalidValue);
    const auto blocks = static_cast<unsigned>((std::uint64_t(table.active_count) + 255) / 256);
    gather_cells<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(table, ranges, output, policy);
    return int(cudaGetLastError());
}
}
