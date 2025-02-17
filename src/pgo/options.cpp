#include <covoxslam/pgo.hpp>

namespace covoxslam::pgo
{
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
