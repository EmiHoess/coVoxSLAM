#pragma once
#include <covoxslam/mapping.hpp>
#include <covoxslam/result.hpp>
#include <span>

namespace covoxslam::pgo
{
enum class backend { native_cpu, cuda_cold };
struct options
{
    bool enabled{};
    backend solver{backend::native_cpu};
    std::size_t maximum_iterations{50};
    double maximum_seconds{30};
    int device{};
    std::uint64_t memory_budget_bytes{1073741824};
};
struct graph_node
{
    std::uint64_t id{};
    pose4 initial;
     bool constant{};
};
bool available(backend solver) noexcept;
}

#if defined(covoxslam_has_pgo)
#include <ceronoceres/ceronoceres.hpp>
namespace covoxslam::pgo
{
struct graph_snapshot
{
    ceronoceres::Problem problem;
    std::vector<std::uint64_t> node_ids;
};
operation_result<graph_snapshot> make_graph(std::span<const graph_node> nodes);
operation_status validate_graph(const graph_snapshot& graph);
operation_result<ceronoceres::PreparedProblem> prepare_graph(graph_snapshot graph);
}
#endif
