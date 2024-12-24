#include "prepared.cuh"
#include "hybrid.cuh"
#include "native.h"

namespace covoxslam::sensor_grid
{
int cuda_sensor_prepared(layout_view view, const float* ranges, prepared_ray* prepared, voxel* output,
    config settings, std::uint32_t ray_count, gather_backend backend, gather_policy policy, void* opaque_stream)
{
    const bool warp = backend == gather_backend::warp_soa_prepared;
    const bool hybrid = backend == gather_backend::warp_soa_hybrid_prepared;

    if ((!warp && !hybrid && backend != gather_backend::thread_interleaved_prepared) || !view.rows || !ray_count ||
        !view.starts || !view.lengths || !view.cells || !view.ray_ids || !view.entry_t || !view.projected ||
        !ranges || !prepared || !output || view.stride != (warp || hybrid ? 1U : 32U) ||
        (hybrid && (!view.row_ids || view.short_rows > view.rows)) ||
        !valid_gather_policy(policy) || !safe_accumulation(settings, ray_count)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    prepare_rays<<<unsigned((std::uint64_t(ray_count) + 255) / 256), 256, 0, stream>>>(ranges, ray_count, settings, prepared);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    if (hybrid)
    {
        if (policy.first_entry_only)
            hybrid_thread_gather_kernel<true><<<(view.rows + 255) / 256, 256, 0, stream>>>(view, 0, view.rows, prepared, settings, policy.max_rays, output);
        else {
            if (view.short_rows) hybrid_thread_gather_kernel<false><<<(view.short_rows + 255) / 256, 256, 0, stream>>>(view, 0, view.short_rows, prepared, settings, policy.max_rays, output);
            const auto long_rows = view.rows - view.short_rows;
            if (long_rows) {
                if (policy.max_rays == 1) hybrid_warp_gather_kernel<true><<<(long_rows + 7) / 8, 256, 0, stream>>>(view, view.short_rows, long_rows, prepared, settings, policy.max_rays, output);
                else hybrid_warp_gather_kernel<false><<<(long_rows + 7) / 8, 256, 0, stream>>>(view, view.short_rows, long_rows, prepared, settings, policy.max_rays, output);
            }
        }
        return int(cudaGetLastError());
    }
    const auto blocks = unsigned((std::uint64_t(view.rows) * (warp ? 32 : 1) + 255) / 256);
    if (warp) prepared_gather<32><<<blocks, 256, 0, stream>>>(view, prepared, settings, policy, output);
    else prepared_gather<1><<<blocks, 256, 0, stream>>>(view, prepared, settings, policy, output);
    return int(cudaGetLastError());
}
}
