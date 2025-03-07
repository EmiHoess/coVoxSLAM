#include <covoxslam/pgo_snapshot.hpp>

namespace covoxslam::pgo
{
operation_result<ceronoceres::SolveResult> solve_cuda(const solve_snapshot& snapshot,
    const options& options, const ceronoceres::NativeLmOptions& lm)
{
    const auto configured = cuda_solve_options(options, lm);
    if (!configured) return std::unexpected(configured.error());
    auto result = ceronoceres::solve(snapshot.prepared(), *configured);
    const auto valid = validate_solve_result(result, snapshot.extents().nodes, configured->backend);

    if (!valid) return std::unexpected(valid.error());
    return result;
}
operation_result<ceronoceres::SolveResult> solve_native(const solve_snapshot& snapshot,
    const options& options, const ceronoceres::NativeLmOptions& lm)
{
    switch (options.solver) {
    case backend::native_cpu: return solve_cpu(snapshot.prepared(), options, lm);
    case backend::cuda_cold: return solve_cuda(snapshot, options, lm);
    }
    return std::unexpected(operation_error{"invalid native solver selection"});
}
}
