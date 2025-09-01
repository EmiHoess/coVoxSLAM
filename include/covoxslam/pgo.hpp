#pragma once
#include <covoxslam/optimization/pose_graph.hpp>
#include <covoxslam/optimization/registration.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <optional>
#include <span>
#include <vector>

#if defined(covoxslam_has_pgo)
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
}
#endif
