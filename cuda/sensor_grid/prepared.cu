#include "prepared.cuh"
#include "native.h"

namespace covoxslam::sensor_grid
{
int cuda_sensor_prepared(layout_view view, const float* ranges, prepared_ray* prepared, voxel* output,
    config settings, std::uint32_t ray_count, gather_backend backend, gather_policy policy, void* opaque_stream)
{
    const bool warp = backend == gather_backend::warp_soa_prepared;

    if ((!warp && backend != gather_backend::thread_interleaved_prepared) || !view.rows || !ray_count ||
        !view.starts || !view.lengths || !view.cells || !view.ray_ids || !view.entry_t || !view.projected ||
        !ranges || !prepared || !output || view.stride != (warp ? 1U : 32U) ||
        !valid_gather_policy(policy) || !safe_accumulation(settings, ray_count)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    prepare_rays<<<unsigned((std::uint64_t(ray_count) + 255) / 256), 256, 0, stream>>>(ranges, ray_count, settings, prepared);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    const auto blocks = unsigned((std::uint64_t(view.rows) * (warp ? 32 : 1) + 255) / 256);
    if (warp) prepared_gather<32><<<blocks, 256, 0, stream>>>(view, prepared, settings, policy, output);
    else prepared_gather<1><<<blocks, 256, 0, stream>>>(view, prepared, settings, policy, output);
    return int(cudaGetLastError());
}
}
