#include <covoxslam/pgo_snapshot.hpp>
#include <cmath>

namespace covoxslam::pgo
{
operation_status validate_snapshot_result(const solve_snapshot& snapshot, const ceronoceres::SolveResult& result)
{
    const auto valid = validate_solve_result(result, snapshot.extents().nodes, result.backend);
    if (!valid) return valid;

    if (result.backend != ceronoceres::Backend::NativeCpu && result.backend != ceronoceres::Backend::NativeCudaCold)
        return std::unexpected(operation_error{"snapshot result did not use a native solver"});
    if (!result.solution_usable) return {};

    for (std::size_t index = 0; index < snapshot.extents().nodes; ++index)
    {
        const auto& node = snapshot.graph().problem.nodes[index];
        if (!node.constant) continue;
        const auto& value = result.poses[index];
        const auto& initial = node.initial;

        if (value.x != initial.x || value.y != initial.y || value.z != initial.z ||
            std::abs(ceronoceres::wrap_yaw(value.yaw - initial.yaw)) > 1.0e-12)
            return std::unexpected(operation_error{"native solver changed a fixed graph pose"});
    }
    const auto& t = result.telemetry;
    const double times[]{t.validation_ms, t.preparation_ms, t.upload_ms, t.evaluation_ms,
        t.assembly_ms, t.linear_solver_ms, t.update_ms, t.synchronization_ms, t.download_ms, t.total_ms};

    for (double time : times)
        if (!std::isfinite(time) || time < 0)
            return std::unexpected(operation_error{"native solver returned invalid timing data"});
    return {};
}
operation_result<ceronoceres::SolveResult> solve_cuda(const solve_snapshot& snapshot,
    const options& options, const ceronoceres::NativeLmOptions& lm)
{
    const auto configured = cuda_solve_options(options, lm);
    if (!configured) return std::unexpected(configured.error());
    auto result = ceronoceres::solve(snapshot.prepared(), *configured);
    const auto valid = validate_snapshot_result(snapshot, result);

    if (!valid) return std::unexpected(valid.error());
    return result;
}
operation_result<ceronoceres::SolveResult> solve_native(const solve_snapshot& snapshot,
    const options& options, const ceronoceres::NativeLmOptions& lm)
{
    operation_result<ceronoceres::SolveResult> result = std::unexpected(operation_error{"invalid native solver selection"});
    switch (options.solver) {
    case backend::native_cpu: result = solve_cpu(snapshot.prepared(), options, lm); break;
    case backend::cuda_cold: result = solve_cuda(snapshot, options, lm); break;
    }
    if (!result) return result;
    const auto valid = validate_snapshot_result(snapshot, *result);
    if (!valid) return std::unexpected(valid.error());
    return result;
}
}
