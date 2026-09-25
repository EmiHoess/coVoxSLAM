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
#include <abseil/abseil.hpp>
namespace covoxslam::pgo
{
operation_result<abseil::cost_breakdown> evaluate_graph(
    const abseil::prepared_problem& problem, std::span<const pose4> poses);
struct equation_statistics
{
    std::size_t active_nodes{}, scalar_dimension{}, sparse_blocks{};
    double maximum_absolute_gradient{};
    abseil::cost_breakdown cost;
    abseil::registration_statistics registration;
};
operation_result<abseil::linearization_snapshot> linearize_graph(
    const abseil::prepared_problem& problem, std::span<const pose4> poses);
operation_status validate_linearization(const abseil::linearization_snapshot& snapshot, std::size_t active_nodes);
operation_result<equation_statistics> summarize_linearization(
    const abseil::linearization_snapshot& snapshot, std::size_t active_nodes);
operation_status validate_lm(const abseil::native_lm_options& value);
operation_result<abseil::solve_options> cpu_solve_options(
    const options& options, const abseil::native_lm_options& lm = {});
operation_result<abseil::solve_result> solve_cpu(const abseil::prepared_problem& problem,
    const options& options, const abseil::native_lm_options& lm = {});
operation_status validate_solve_result(const abseil::solve_result& result,
    std::size_t nodes, abseil::backend expected_backend);
operation_result<std::vector<pose4>> solution_poses(const abseil::solve_result& result, std::size_t nodes);
struct solve_statistics
{
    abseil::termination termination{abseil::termination::numerical_breakdown};
    bool usable{}, converged{};
    std::size_t accepted_steps{}, rejected_steps{}, linear_iterations{};
    abseil::cost_breakdown initial_cost, final_cost;
};
operation_result<solve_statistics> summarize_solve(const abseil::solve_result& result);
struct residual_report
{
    abseil::backend solver{abseil::backend::native_cpu};
    abseil::cost_breakdown initial, final;
    abseil::registration_statistics registration;
    bool usable{};
};
operation_result<residual_report> report_residuals(const abseil::solve_result& result);
struct residual_comparison
{
    abseil::cost_breakdown final_difference;
    bool same_admission{};
};
operation_result<residual_comparison> compare_residuals(const residual_report& cpu, const residual_report& cuda);
operation_status write_solver_report(std::ostream& output, const abseil::solve_result& result);
struct linear_options
{
    std::size_t maximum_iterations{256};
    double tolerance{1.0e-8};
};
operation_result<abseil::native_lm_options> configure_linear(
    linear_options linear, abseil::native_lm_options lm = {});
operation_status write_iteration_report(std::ostream& output, const abseil::solve_result& result);
struct snapshot_extents
{
    std::size_t nodes{}, relative_factors{}, absolute_factors{}, groups{}, samples{};
    std::size_t tsdf_tiles{}, esdf_tiles{};
     std::uint64_t record_bytes{};
};
class solve_snapshot
{
    std::shared_ptr<const graph_snapshot> input_;
    abseil::prepared_problem prepared_;
    snapshot_extents extents_;
    solve_snapshot(std::shared_ptr<const graph_snapshot> input,
        abseil::prepared_problem prepared, snapshot_extents extents)
        : input_(std::move(input)), prepared_(std::move(prepared)), extents_(extents) {}
    friend operation_result<solve_snapshot> freeze_graph(graph_snapshot graph);
    friend operation_result<solve_snapshot> freeze_field_graph(graph_snapshot,
        std::shared_ptr<const abseil::field_snapshot>);
public:
    const graph_snapshot& graph() const noexcept { return *input_; }
    const abseil::prepared_problem& prepared() const noexcept { return prepared_; }
    const snapshot_extents& extents() const noexcept { return extents_; }
    const std::string& hash() const noexcept { return prepared_.hash(); }
};
operation_result<snapshot_extents> measure_snapshot(const graph_snapshot& graph);
operation_result<solve_snapshot> freeze_graph(graph_snapshot graph);
operation_result<solve_snapshot> freeze_field_graph(graph_snapshot graph,
    std::shared_ptr<const abseil::field_snapshot> fields);
operation_result<abseil::solve_options> cuda_solve_options(
    const options& options, const abseil::native_lm_options& lm = {});
operation_result<abseil::solver_telemetry> cuda_telemetry(const abseil::solve_result& result);
operation_result<abseil::solve_result> solve_cuda(const solve_snapshot& snapshot,
    const options& options, const abseil::native_lm_options& lm = {});
operation_result<abseil::solve_result> solve_native(const solve_snapshot& snapshot,
    const options& options, const abseil::native_lm_options& lm = {});
operation_status validate_snapshot_result(const solve_snapshot& snapshot, const abseil::solve_result& result);
}
#endif
