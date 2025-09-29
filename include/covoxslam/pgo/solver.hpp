#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/pgo/pose_graph.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#if defined(covoxslam_has_pgo)
#include <ceronoceres/ceronoceres.hpp>
namespace covoxslam::pgo
{
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
struct snapshot_extents
{
    std::size_t nodes{}, relative_factors{}, absolute_factors{}, groups{}, samples{};
    std::size_t tsdf_tiles{}, esdf_tiles{};
     std::uint64_t record_bytes{};
};
class solve_snapshot
{
    std::shared_ptr<const graph_snapshot> input_;
    ceronoceres::PreparedProblem prepared_;
    snapshot_extents extents_;
    solve_snapshot(std::shared_ptr<const graph_snapshot> input,
        ceronoceres::PreparedProblem prepared, snapshot_extents extents)
        : input_(std::move(input)), prepared_(std::move(prepared)), extents_(extents) {}
    friend operation_result<solve_snapshot> freeze_graph(graph_snapshot graph);
    friend operation_result<solve_snapshot> freeze_field_graph(graph_snapshot,
        std::shared_ptr<const ceronoceres::field_snapshot>);
public:
    const graph_snapshot& graph() const noexcept { return *input_; }
    const ceronoceres::PreparedProblem& prepared() const noexcept { return prepared_; }
    const snapshot_extents& extents() const noexcept { return extents_; }
    const std::string& hash() const noexcept { return prepared_.hash(); }
};
operation_result<snapshot_extents> measure_snapshot(const graph_snapshot& graph);
operation_result<solve_snapshot> freeze_graph(graph_snapshot graph);
operation_result<solve_snapshot> freeze_field_graph(graph_snapshot graph,
    std::shared_ptr<const ceronoceres::field_snapshot> fields);
operation_result<ceronoceres::SolveOptions> cuda_solve_options(
    const options& options, const ceronoceres::NativeLmOptions& lm = {});
operation_result<ceronoceres::SolverTelemetry> cuda_telemetry(const ceronoceres::SolveResult& result);
operation_result<ceronoceres::SolveResult> solve_cuda(const solve_snapshot& snapshot,
    const options& options, const ceronoceres::NativeLmOptions& lm = {});
operation_result<ceronoceres::SolveResult> solve_native(const solve_snapshot& snapshot,
    const options& options, const ceronoceres::NativeLmOptions& lm = {});
operation_status validate_snapshot_result(const solve_snapshot& snapshot, const ceronoceres::SolveResult& result);
}
#endif
