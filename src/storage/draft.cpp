#include <covoxslam/storage/draft.hpp>

namespace covoxslam::storage
{
#if defined(covoxslam_has_urd)
template class draft<schema::tsdf, layout::pool_soa>;
template class draft<schema::esdf, layout::pool_soa>;
template class draft<schema::blocks_esdf, layout::pool_soa>;
template class draft<schema::projective_esdf, layout::pool_soa>;
#endif
}
