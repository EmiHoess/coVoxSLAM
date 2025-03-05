#pragma once
#include <covoxslam/mapping.hpp>
#include <covoxslam/result.hpp>
#include <span>
#include <iosfwd>

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
struct surface_bounds
{
    std::array<double, 3> minimum{}, maximum{};
    bool valid{};
};
struct surface_sample {
    ceronoceres::Vec3d point;
    double distance{}, weight{};
     std::uint64_t ordinal{};
};
struct directed_candidate
{
    std::size_t source{}, target{};
    std::uint64_t diagnostic_id{};
    std::vector<surface_sample> samples;
};
struct candidate_options
{
    std::size_t sample_stride{16};
    double overlap_padding_m{0.6};
};
struct candidate_set
{
    std::vector<surface_bounds> bounds;
    std::vector<directed_candidate> directions;
};
operation_result<candidate_set> find_candidates(std::span<const submap> maps, candidate_options options = {});
operation_status validate_candidates(const candidate_set& candidates, std::size_t map_count);
struct candidate_statistics
{
    std::size_t surface_maps{}, overlap_pairs{}, directed_groups{}, samples{};
};
operation_result<candidate_statistics> summarize_candidates(const candidate_set& candidates);
struct registration_options
{
    ceronoceres::Loss loss{ceronoceres::LossKind::Huber, 0.20};
    double minimum_weight{1.0e-6};
    double maximum_weight{10000.0};
    double minimum_gradient{1.0e-6};
    double maximum_residual_m{1.2};
    bool nearest_fallback{};
};
operation_status append_registration(graph_snapshot& graph, std::span<const submap> maps,
    const candidate_set& candidates, registration_options options = {});
operation_result<ceronoceres::CostBreakdown> evaluate_graph(
    const ceronoceres::PreparedProblem& problem, std::span<const pose4> poses);
struct equation_statistics
{
    std::size_t active_nodes{}, scalar_dimension{}, sparse_blocks{};
    double maximum_absolute_gradient{};
    ceronoceres::CostBreakdown cost;
    ceronoceres::RegistrationStatistics registration;
};
operation_result<ceronoceres::LinearizationSnapshot> linearize_graph(
    const ceronoceres::PreparedProblem& problem, std::span<const pose4> poses);
operation_status validate_linearization(const ceronoceres::LinearizationSnapshot& snapshot, std::size_t active_nodes);
operation_result<equation_statistics> summarize_linearization(
    const ceronoceres::LinearizationSnapshot& snapshot, std::size_t active_nodes);
operation_status validate_lm(const ceronoceres::NativeLmOptions& value);
operation_result<ceronoceres::SolveOptions> cpu_solve_options(
    const options& options, const ceronoceres::NativeLmOptions& lm = {});
operation_result<ceronoceres::SolveResult> solve_cpu(const ceronoceres::PreparedProblem& problem,
    const options& options, const ceronoceres::NativeLmOptions& lm = {});
operation_status validate_solve_result(const ceronoceres::SolveResult& result,
    std::size_t nodes, ceronoceres::Backend expected_backend);
operation_result<std::vector<pose4>> solution_poses(const ceronoceres::SolveResult& result, std::size_t nodes);
struct solve_statistics
{
    ceronoceres::Termination termination{ceronoceres::Termination::BackendFailure};
    bool usable{}, converged{};
    std::size_t accepted_steps{}, rejected_steps{}, linear_iterations{};
    ceronoceres::CostBreakdown initial_cost, final_cost;
};
operation_result<solve_statistics> summarize_solve(const ceronoceres::SolveResult& result);
struct residual_report
{
    ceronoceres::Backend solver{ceronoceres::Backend::NativeCpu};
    ceronoceres::CostBreakdown initial, final;
    ceronoceres::RegistrationStatistics registration;
    bool usable{};
};
operation_result<residual_report> report_residuals(const ceronoceres::SolveResult& result);
struct residual_comparison
{
    ceronoceres::CostBreakdown final_difference;
    bool same_admission{};
};
operation_result<residual_comparison> compare_residuals(const residual_report& cpu, const residual_report& cuda);
operation_status write_solver_report(std::ostream& output, const ceronoceres::SolveResult& result);
struct linear_options
{
    std::size_t maximum_iterations{256};
    double tolerance{1.0e-8};
};
operation_result<ceronoceres::NativeLmOptions> configure_linear(
    linear_options linear, ceronoceres::NativeLmOptions lm = {});
operation_status write_iteration_report(std::ostream& output, const ceronoceres::SolveResult& result);
operation_result<graph_snapshot> make_graph(std::span<const graph_node> nodes);
operation_status validate_graph(const graph_snapshot& graph);
operation_result<ceronoceres::PreparedProblem> prepare_graph(graph_snapshot graph);
operation_result<std::optional<graph_snapshot>> make_graph_if_enabled(
    std::span<const graph_node> nodes, const options& options);
}
#endif
