#include "diagnostics_internal.hpp"
#include <algorithm>

namespace covoxslam::pgo
{
operation_status validate_lm(const ceronoceres::NativeLmOptions& value)
{
    const double positive[]{value.initial_damping, value.minimum_damping, value.maximum_damping,
        value.maximum_translation_step_m, value.maximum_yaw_step_rad, value.pcg_tolerance};
    const double nonnegative[]{value.minimum_cost_change, value.minimum_gradient_norm, value.minimum_update_norm};

    if (!value.maximum_rejected_steps || value.maximum_rejected_steps > 1000000 ||
        !value.pcg_maximum_iterations || value.pcg_maximum_iterations > 1000000 ||
        value.initial_damping < value.minimum_damping || value.maximum_damping < value.initial_damping ||
        !std::all_of(std::begin(positive), std::end(positive), [](double x) { return std::isfinite(x) && x > 0; }) ||
        !std::all_of(std::begin(nonnegative), std::end(nonnegative), [](double x) { return std::isfinite(x) && x >= 0; }))
        return std::unexpected(operation_error{"invalid native LM settings"});
    return {};
}
operation_result<ceronoceres::SolveOptions> cpu_solve_options(
    const options& options, const ceronoceres::NativeLmOptions& lm)
{
    const auto valid = validate_options(options);
    if (!valid) return std::unexpected(valid.error());
    const auto valid_lm = validate_lm(lm);
    if (!valid_lm) return std::unexpected(valid_lm.error());

    if (!options.enabled || options.solver != backend::native_cpu)
        return std::unexpected(operation_error{"native CPU solve requires enabled CPU settings"});
    ceronoceres::SolveOptions result;
    result.backend = ceronoceres::Backend::NativeCpu;
    result.common = {options.maximum_iterations, options.maximum_seconds};
    result.backend_options = ceronoceres::NativeCpuOptions{lm};
    return result;
}
operation_result<ceronoceres::SolveResult> solve_cpu(const ceronoceres::PreparedProblem& problem,
    const options& options, const ceronoceres::NativeLmOptions& lm)
{
    auto configured = cpu_solve_options(options, lm);
    if (!configured) return std::unexpected(configured.error());

    if (problem.empty()) return std::unexpected(operation_error{"cannot solve an empty prepared graph"});
    return ceronoceres::solve(problem, *configured);
}
}
