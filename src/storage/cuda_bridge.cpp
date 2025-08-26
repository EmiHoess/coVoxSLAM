#include <covoxslam/storage/device_buffer.hpp>

namespace covoxslam::storage
{
#if defined(covoxslam_has_urd)
template class device_buffer<schema::tsdf, layout::pool_soa>;
template class device_buffer<schema::esdf, layout::pool_soa>;
template class device_buffer<schema::projective_esdf, layout::pool_soa>;
#endif
}
