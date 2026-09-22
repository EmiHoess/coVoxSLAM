#pragma once
#include <covoxslam/core/byte_units.hpp>
#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/pipeline/submap.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace covoxslam::pgo
{
enum class backend { native_cpu, cuda_cold };
inline constexpr std::uint64_t default_solver_memory_bytes = bytes_per_gibibyte;
struct options
{
    bool enabled{};
    backend solver{backend::native_cpu};
    std::size_t maximum_iterations{50};
    double maximum_seconds{30};
    int device{};
    std::uint64_t memory_budget_bytes{default_solver_memory_bytes};
};
struct graph_node
{
    std::uint64_t id{};
    pose4 initial;
     bool constant{};
};
bool available(backend solver) noexcept;
operation_status validate_options(const options& value);
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
struct relative_measurement
{
    std::uint64_t source_id{}, target_id{};
    pose4 measured;
    ceronoceres::Matrix4 sqrt_information{ceronoceres::identity_matrix4()};
    ceronoceres::Loss loss{ceronoceres::LossKind::Huber, 1.0};
     std::uint64_t diagnostic_id{};
};
operation_status append_relative(graph_snapshot& graph, std::span<const relative_measurement> measurements);
operation_status append_odometry(graph_snapshot& graph, std::span<const submap> maps);
struct absolute_measurement
{
    std::uint64_t node_id{};
    std::optional<pose4> world_pose;
    ceronoceres::Matrix4 sqrt_information{ceronoceres::identity_matrix4()};
    ceronoceres::Loss loss{ceronoceres::LossKind::Huber, 1.0};
     std::uint64_t diagnostic_id{};
};
operation_status append_absolute(graph_snapshot& graph, std::uint64_t reference_id,
    pose4 reference_world, std::span<const absolute_measurement> measurements);
operation_result<graph_snapshot> make_submap_graph(std::span<const submap> maps,
    std::span<const absolute_measurement> measurements = {});
operation_result<graph_snapshot> make_graph(std::span<const graph_node> nodes);
operation_status validate_graph(const graph_snapshot& graph);
operation_result<ceronoceres::PreparedProblem> prepare_graph(graph_snapshot graph);
operation_result<std::optional<graph_snapshot>> make_graph_if_enabled(
    std::span<const graph_node> nodes, const options& options);
}
#endif
