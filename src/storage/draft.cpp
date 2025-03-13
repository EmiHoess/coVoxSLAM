#include <covoxslam/storage/selection.hpp>

namespace covoxslam::storage
{
template class draft<schema::tsdf, layout::pooled_aos>;
template class draft<schema::esdf, layout::pooled_aos>;
template class draft<schema::blocks_esdf, layout::pooled_aos>;
template class draft<schema::projective_esdf, layout::pooled_aos>;
#if defined(covoxslam_has_urd)
template class draft<schema::tsdf, layout::tile_aosoa>;
template class draft<schema::esdf, layout::tile_aosoa>;
template class draft<schema::blocks_esdf, layout::tile_aosoa>;
template class draft<schema::projective_esdf, layout::tile_aosoa>;
template class draft<schema::tsdf, layout::pool_soa>;
template class draft<schema::esdf, layout::pool_soa>;
template class draft<schema::blocks_esdf, layout::pool_soa>;
template class draft<schema::projective_esdf, layout::pool_soa>;
#endif
}
