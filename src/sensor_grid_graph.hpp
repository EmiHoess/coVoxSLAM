#pragma once
#include <covoxslam/sensor_grid_layout_types.h>
#include <covoxslam/sensor_grid_precision.h>
#include <cuda_runtime_api.h>
#include <array>

namespace covoxslam::sensor_grid
{
struct graph_key
{
    gather_backend backend{};
    std::uint32_t cap{}, precision{accumulation_bits};
    std::uint64_t generation{};
    bool first{}, profile{};
    std::array<const void*, 5> buffers{};
    bool operator==(const graph_key&) const = default;
};
struct graph_plan
{
    graph_key key{};
    cudaGraph_t graph{};
    cudaGraphExec_t executable{};
    void release() noexcept
    {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        graph = nullptr;
         executable = nullptr;
    }
};
}
