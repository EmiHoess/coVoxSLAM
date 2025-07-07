#include <covoxslam/storage/cuda_draft.hpp>

namespace covoxslam::storage
{
#if defined(covoxslam_has_urd)
template class cuda_draft<schema::tsdf, layout::pool_soa>;
template class cuda_draft<schema::esdf, layout::pool_soa>;
template class cuda_draft<schema::blocks_esdf, layout::pool_soa>;
template class cuda_draft<schema::projective_esdf, layout::pool_soa>;
#endif
}
