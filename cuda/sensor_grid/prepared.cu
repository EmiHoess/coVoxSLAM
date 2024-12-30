#include "prepared.cuh"
#include "hybrid.cuh"
#include "subwarp.cuh"
#include "memory_layout.cuh"
#include "compact.cuh"
#include "native.h"

namespace covoxslam::sensor_grid
{
int cuda_sensor_compact(layout_view view, const float* ranges, compact_ray* prepared, voxel* output,
    config settings, std::uint32_t ray_count, gather_policy policy, void* opaque_stream)
{
    if (!view.rows || !ray_count || !view.starts || !view.lengths || !view.cells || !view.row_ids ||
        !view.ray_ids || !view.entry_t || !view.projected || !ranges || !prepared || !output ||
        view.stride != 1 || view.short_rows > view.rows || !valid_gather_policy(policy)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    prepare_compact<<<unsigned((std::uint64_t(ray_count) + 255) / 256), 256, 0, stream>>>(ranges, ray_count, settings, prepared);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);

    if (policy.first_entry_only) compact_thread_gather<true><<<(view.rows + 255) / 256, 256, 0, stream>>>(view, 0, view.rows, prepared, settings, policy.max_rays, output);
    else {
        if (view.short_rows) compact_thread_gather<false><<<(view.short_rows + 255) / 256, 256, 0, stream>>>(view, 0, view.short_rows, prepared, settings, policy.max_rays, output);
        const auto count = view.rows - view.short_rows;
        if (count) {
            if (policy.max_rays == 1) compact_warp_gather<true><<<(count + 7) / 8, 256, 0, stream>>>(view, view.short_rows, count, prepared, settings, policy.max_rays, output);
            else compact_warp_gather<false><<<(count + 7) / 8, 256, 0, stream>>>(view, view.short_rows, count, prepared, settings, policy.max_rays, output);
        }
    }
    return int(cudaGetLastError());
}
int cuda_sensor_prepared(layout_view view, const float* ranges, prepared_ray* prepared, voxel* output,
    config settings, std::uint32_t ray_count, gather_backend backend, gather_policy policy, void* opaque_stream,
    void* profile_events)
{
    const bool warp = backend == gather_backend::warp_soa_prepared;
    const bool hybrid = backend == gather_backend::warp_soa_hybrid_prepared;
    const bool subwarp = backend == gather_backend::warp_soa_subwarp_prepared;
    const bool packed = backend >= gather_backend::hybrid_short_interleaved && backend <= gather_backend::hybrid_ray16;

    if ((!warp && !hybrid && !subwarp && !packed && backend != gather_backend::thread_interleaved_prepared) || !view.rows || !ray_count ||
        view.rows > 16777216 || view.rows > view.entries || ray_count > 1048576 ||
        !view.starts || !view.lengths || !view.cells || (!view.ray_ids && !(packed && view.ray_ids16)) || !view.entry_t || !view.projected ||
        !ranges || !prepared || !output || view.stride != (warp || hybrid || subwarp || packed ? 1U : 32U) ||
        (packed && (!view.ranges || !view.row_ids || view.short_rows > view.rows || (view.short_stride != 1 && view.short_stride != 32))) ||
        (subwarp && !view.row_ids) ||
        (hybrid && (!view.row_ids || view.short_rows > view.rows)) ||
        !valid_gather_policy(policy) || !safe_accumulation(settings, ray_count)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    prepare_rays<<<unsigned((std::uint64_t(ray_count) + 255) / 256), 256, 0, stream>>>(ranges, ray_count, settings, prepared);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    if (packed) return view.ray_ids16
        ? launch_memory_layout<true>(view, prepared, output, settings, policy, stream)
        : launch_memory_layout<false>(view, prepared, output, settings, policy, stream);
    if (subwarp)
    {
        auto* events = static_cast<cudaEvent_t*>(profile_events);
        if (events) {
            const auto recorded = cudaEventRecord(events[0], stream);
            if (recorded != cudaSuccess) return int(recorded);
        }
        if (policy.first_entry_only) {
            subwarp_thread_gather_kernel<subwarp_selection::first_entry><<<(view.rows + 255) / 256, 256, 0, stream>>>(view, 0, view.rows, prepared, settings, policy.max_rays, output);
            const auto launched = cudaGetLastError();
            if (launched != cudaSuccess) return int(launched);
            if (events)
                for (unsigned i = 1; i < 5; ++i) {
                    const auto recorded = cudaEventRecord(events[i], stream);
                    if (recorded != cudaSuccess) return int(recorded);
                }
            return 0;
        }
        if (!policy.max_rays) return launch_subwarp<subwarp_selection::all>(view, prepared, output, settings, policy, stream, events);
        if (policy.max_rays == 1) return launch_subwarp<subwarp_selection::cap_one>(view, prepared, output, settings, policy, stream, events);
        return launch_subwarp<subwarp_selection::capped>(view, prepared, output, settings, policy, stream, events);
    }
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
