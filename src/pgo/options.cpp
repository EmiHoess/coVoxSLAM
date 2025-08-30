#include <covoxslam/optimization/pose_graph.hpp>
#include <cmath>

namespace covoxslam::pgo
{
operation_status validate_options(const options& value)
{
    if (!value.enabled) return {};
    if (!available(value.solver)) return std::unexpected(operation_error{"selected native PGO backend is unavailable"});
    if (!value.maximum_iterations || value.maximum_iterations > 1000000 ||
        !std::isfinite(value.maximum_seconds) || value.maximum_seconds <= 0 || value.device < 0)
        return std::unexpected(operation_error{"invalid PGO execution limits"});
    return {};
}
bool available(backend solver) noexcept
{
#if defined(covoxslam_has_pgo)
    switch (solver) {
    case backend::native_cpu: return ceronoceres::backend_available(ceronoceres::Backend::NativeCpu);
    case backend::cuda_cold: return ceronoceres::backend_available(ceronoceres::Backend::NativeCudaCold);
    }
#else
    static_cast<void>(solver);
#endif
    return false;
}
}
