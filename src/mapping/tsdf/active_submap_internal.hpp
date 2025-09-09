#pragma once
#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/mapping/tsdf/active_submap.hpp>
#include "mapping/tsdf/fusion.cuh"
#include "mapping/tsdf/discovery.cuh"
#include "mapping/tsdf/allocation.cuh"
#include <cuda_runtime_api.h>
#include <array>
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
    std::array<cudaEvent_t, 8> events{};
    std::array<void*, 20> allocations{};
    std::size_t allocated{};
    bool selected{}, failed{};
    operation_result<transfer_times> integrate_resident(field_view source,
        detail::cuda_projective_pose transform, interpolation mode);
    operation_result<tsdf_tiles> download_soa();
    operation_status check(int code)
    {
        if (!code) return {};
        failed = true;
        return std::unexpected(operation_error{cudaGetErrorString(static_cast<cudaError_t>(code))});
    }
    template<class value_type>
    operation_status allocate(value_type*& pointer, std::size_t count) {
        const auto result = check(cudaMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(value_type)));
        if (pointer) allocations[allocated++] = pointer;
        return result;
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

        for (auto event : events)
            if (event) cudaEventDestroy(event);

        for (std::size_t i = 0; i < allocated; ++i) cudaFree(allocations[i]);
        if (stream) cudaStreamDestroy(stream);
    }
};
}
