#include "combined.cuh"
#include "native.h"

namespace covoxslam::sensor_grid
{
int cuda_sensor_combined(layout_view view, const float* ranges, compact_ray* prepared, voxel* output,
    config settings, std::uint32_t ray_count, gather_policy policy, void* opaque_stream)
{
    if (!view.rows || view.rows > 16777216 || view.rows > view.entries || !ray_count || ray_count > 1048576 ||
        !view.starts || !view.lengths || !view.cells || !view.row_ids ||
        (!view.ray_ids && !view.ray_ids16) || (view.ray_ids16 && ray_count > 65536) ||
        !view.entry_t || !view.projected || !ranges || !prepared || !output ||
        view.stride != 1 || view.short_rows > view.rows || view.block_rows > view.rows - view.short_rows ||
        !valid_gather_policy(policy) || !safe_accumulation(settings, ray_count) ||
        !(settings.voxel_size > 0) || !(settings.truncation > settings.voxel_size) ||
        !(settings.radius > 0) || !std::isfinite(settings.radius)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    prepare_compact<<<(ray_count + 255) / 256, 256, 0, stream>>>(ranges, ray_count, settings, prepared);
    const auto error = cudaGetLastError();

    if (error != cudaSuccess) return int(error);
    return view.ray_ids16 ? launch_combined<true>(view, prepared, output, settings, policy, stream)
        : launch_combined<false>(view, prepared, output, settings, policy, stream);
}
}
