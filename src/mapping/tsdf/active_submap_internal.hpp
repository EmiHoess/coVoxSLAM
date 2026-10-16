#pragma once
#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/mapping/tsdf/active_submap.hpp>
#include "mapping/tsdf/fusion.cuh"
#include "mapping/tsdf/discovery.cuh"
#include "mapping/tsdf/allocation.cuh"
#include <cuda_runtime_api.h>
#if defined(covoxslam_has_urd)
#include <covoxslam/storage/device_buffer.hpp>
#endif

namespace covoxslam::sensor_grid
{
struct gpu_map::implementation
{
    config settings{};
    geometry grid{};
    int device{};
     std::uint32_t capacity{};
    bool soa_selected{};
#if defined(covoxslam_has_urd)
    std::unique_ptr<storage::device_buffer<storage::schema::tsdf, storage::layout::pool_soa>> soa;
#endif
    detail::cuda_tile_coordinate* keys{};
    std::uint32_t* slots{};
    std::uint8_t* mask{};
    discovery_view discovery{};
    resident_discovery_view resident{};
    transfer_options options{};
    unsigned long long* updates{};
    std::vector<int3> indices;
    std::unordered_map<int3, std::uint32_t, int3_hash> index;
    cudaStream_t stream{};
    cudaEvent_t mask_start{}, mask_end{};
    cudaEvent_t fusion_start{}, fusion_end{};
    cudaEvent_t allocation_start{}, allocation_end{};
    cudaEvent_t discovery_start{}, discovery_end{};
    bool selected{}, failed{};
    operation_result<transfer_times> integrate_resident(field_view source,
        detail::cuda_projective_pose transform, interpolation mode,
        std::uint64_t kernel_launches);
    operation_result<tsdf_tiles> download_soa();
    operation_status check(int code)
    {
        if (!code) return {};
        failed = true;
        return std::unexpected(operation_error{cudaGetErrorString(static_cast<cudaError_t>(code))});
    }
    template<class value_type>
    operation_status allocate(value_type*& pointer, std::size_t count) {
        return check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(value_type)));
    }
    operation_status wait(operation_status status)
    {
        const auto result = check(cudaStreamSynchronize(stream));
        return status ? result : status;
    }
    operation_status reset_payload()
    {
        if (!soa) return std::unexpected(operation_error{"resident SoA pool is unavailable"});
        auto descriptor = soa->descriptor();
        if (!descriptor) return std::unexpected(descriptor.error());
        return check(cuda_sensor_reset_soa(resident, *descriptor, stream));
    }
    ~implementation()
    {
        if (!selected) return;
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);

        if (mask_start) cudaEventDestroy(mask_start);
        if (mask_end) cudaEventDestroy(mask_end);
        if (fusion_start) cudaEventDestroy(fusion_start);
        if (fusion_end) cudaEventDestroy(fusion_end);
        if (allocation_start) cudaEventDestroy(allocation_start);
        if (allocation_end) cudaEventDestroy(allocation_end);
        if (discovery_start) cudaEventDestroy(discovery_start);
        if (discovery_end) cudaEventDestroy(discovery_end);
        if (keys) cudaFree(keys);
        if (slots) cudaFree(slots);
        if (mask) cudaFree(mask);
        if (updates) cudaFree(updates);
        if (discovery.hash) cudaFree(discovery.hash);
        if (discovery.keys) cudaFree(discovery.keys);
        if (discovery.state) cudaFree(discovery.state);
        if (resident.hash) cudaFree(resident.hash);
        if (resident.scratch_keys) cudaFree(resident.scratch_keys);
        if (resident.offsets) cudaFree(resident.offsets);
        if (resident.sums) cudaFree(resident.sums);
        if (resident.missing) cudaFree(resident.missing);
        if (stream) cudaStreamDestroy(stream);
    }
};
}
