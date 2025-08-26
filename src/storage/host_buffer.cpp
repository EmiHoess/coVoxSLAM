#include <covoxslam/storage/host_buffer.hpp>

namespace covoxslam::storage
{
#if defined(covoxslam_has_urd)
template class host_buffer<schema::tsdf, layout::pool_soa>;
template class host_buffer<schema::esdf, layout::pool_soa>;
template class host_buffer<schema::projective_esdf, layout::pool_soa>;
#endif
}
