#include <covoxslam/optimization/solver.hpp>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace covoxslam::pgo
{
operation_status write_iteration_report(std::ostream& output, const ceronoceres::SolveResult& result)
{
    const auto summary = summarize_solve(result);
    if (!summary) return std::unexpected(summary.error());
    std::ostringstream text;
    text << std::setprecision(17) << "iteration accepted before after damping gradient update linear_iterations seconds linear_seconds\n";

    for (const auto& step : result.iterations)
    {
        if (!std::isfinite(step.objective_before) || step.objective_before < 0 ||
            (step.accepted && (!std::isfinite(step.objective_after) || step.objective_after < 0)) ||
            !std::isfinite(step.damping) || step.damping <= 0 ||
            !std::isfinite(step.iteration_seconds) || step.iteration_seconds < 0 ||
            !std::isfinite(step.linear_solver_seconds) || step.linear_solver_seconds < 0)
            return std::unexpected(operation_error{"invalid reported native iteration"});
        text << step.iteration << ' ' << step.accepted << ' ' << step.objective_before << ' '
            << step.objective_after << ' ' << step.damping << ' ' << step.gradient_norm << ' '
            << step.update_norm << ' ' << step.linear_iterations << ' ' << step.iteration_seconds << ' '
            << step.linear_solver_seconds << '\n';
    }
    output << text.str();
    if (!output) return std::unexpected(operation_error{"cannot write native iteration diagnostics"});
    return {};
}
operation_status write_solver_report(std::ostream& output, const ceronoceres::SolveResult& result)
{
    const auto residuals = report_residuals(result);
    if (!residuals) return std::unexpected(residuals.error());
    const auto summary = summarize_solve(result);
    if (!summary) return std::unexpected(summary.error());
    const auto& t = result.telemetry;
    std::ostringstream text;
    text << std::setprecision(17)
        << "solver " << ceronoceres::backend_name(result.backend) << '\n'
        << "termination " << ceronoceres::termination_name(result.termination) << '\n'
        << "usable " << summary->usable << " converged " << summary->converged << '\n'
        << "relative_cost " << residuals->initial.relative << ' ' << residuals->final.relative << '\n'
        << "absolute_cost " << residuals->initial.absolute << ' ' << residuals->final.absolute << '\n'
        << "registration_cost " << residuals->initial.registration << ' ' << residuals->final.registration << '\n'
        << "registration_rows " << residuals->registration.accepted_rows << '\n'
        << "accepted_steps " << summary->accepted_steps << " rejected_steps " << summary->rejected_steps << '\n'
        << "linear_iterations " << summary->linear_iterations << '\n'
        << "validation_ms " << t.validation_ms << " preparation_ms " << t.preparation_ms << '\n'
        << "upload_ms " << t.upload_ms << " evaluation_ms " << t.evaluation_ms << '\n'
        << "assembly_ms " << t.assembly_ms << " linear_solver_ms " << t.linear_solver_ms << '\n'
        << "update_ms " << t.update_ms << " synchronization_ms " << t.synchronization_ms << '\n'
        << "download_ms " << t.download_ms << " total_ms " << t.total_ms << '\n'
        << "upload_bytes " << t.host_to_device_bytes << " download_bytes " << t.device_to_host_bytes << '\n'
        << "peak_device_bytes " << t.peak_device_bytes << " synchronizations " << t.synchronization_count << '\n';

    for (std::size_t index = 0; index < residuals->registration.rejections.size(); ++index)
        text << "registration_rejection " << index << ' ' << residuals->registration.rejections[index] << '\n';
    output << text.str();

    if (!output) return std::unexpected(operation_error{"cannot write solver diagnostics"});
    return {};
}
}
