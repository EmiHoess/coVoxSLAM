#pragma once
#include <covoxslam/sensor_grid_map.hpp>
#include "sensor_grid/map_native.h"
#include "sensor_grid/candidate_list.h"
#include "sensor_grid/discovery.h"
#include <cuda_runtime_api.h>
#include <array>

namespace covoxslam::sensor_grid
{
struct gpu_map::implementation
{
    config settings{};
    geometry grid{};
    int device{};
     std::uint32_t capacity{};
    detail::cuda_tsdf_tile* tiles{};
    detail::cuda_tile_coordinate* keys{};
    std::uint32_t* slots{};
    std::uint8_t* mask{};
    std::uint32_t* mask_offsets{}, *mask_cells{};
    std::uint32_t* candidate_masks{}, *active_cells{};
    std::uint32_t active_count{};
    unsigned long long* candidate_count{};
    candidate_list_view candidate_list{};
    discovery_view discovery{};
    transfer_options options{};
    unsigned long long* updates{};
    std::vector<std::uint8_t> flags;
    std::vector<int3> indices;
    std::unordered_map<int3, std::uint32_t, int3_hash> index;
    cudaStream_t stream{};
    std::array<cudaEvent_t, 8> events{};
    std::array<void*, 15> allocations{};
    std::size_t allocated{};
    bool selected{}, failed{};
    operation_result<std::vector<int3>> discover_gpu(field_view source,
        detail::cuda_projective_pose transform, float& elapsed);
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
