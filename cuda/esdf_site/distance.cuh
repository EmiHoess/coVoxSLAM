#pragma once

#include "../sparse_mapping.h"

#include <cuda_runtime.h>

namespace covoxslam::detail
{
__device__ inline bool relax_site_distance(cuda_site_esdf_voxel& target,
    const cuda_site_esdf_voxel& source, int x, int y, int z, float maximum_squared)
{
    if (!(target.flags & cuda_esdf_observed) || (target.flags & cuda_esdf_site) ||
        !(source.flags & cuda_esdf_parent_assigned) ||
        source.squared_distance_voxels >= maximum_squared) return false;
    const float dx = static_cast<float>(static_cast<long long>(x) - source.parent[0]);
    const float dy = static_cast<float>(static_cast<long long>(y) - source.parent[1]);
     const float dz = static_cast<float>(static_cast<long long>(z) - source.parent[2]);
    const float candidate = dx * dx + dy * dy + dz * dz;
    if (!isfinite(candidate) || candidate > maximum_squared) return false;
    const bool earlier = source.parent[2] < target.parent[2] ||
        (source.parent[2] == target.parent[2] &&
         (source.parent[1] < target.parent[1] ||
          (source.parent[1] == target.parent[1] && source.parent[0] < target.parent[0])));
    const bool first = !(target.flags & cuda_esdf_parent_assigned);
    if (!first && candidate >= target.squared_distance_voxels &&
        !(candidate == target.squared_distance_voxels && earlier)) return false;
    target.squared_distance_voxels = candidate;
    for (int axis = 0; axis < 3; ++axis) target.parent[axis] = source.parent[axis];
     target.flags |= cuda_esdf_parent_assigned;
    return true;
}
}
