#pragma once
#include "../projective/geometry.cuh"
#include <covoxslam/sensor_grid_sampling.h>

namespace covoxslam::sensor_grid
{
struct affine_transfer { float3 axes[3]; };
inline affine_transfer prepare_affine(detail::cuda_projective_pose transform, float size)
{
    transform.translation[0] = transform.translation[1] = transform.translation[2] = 0;
    return {{detail::projective_reference_point(transform, {size, 0, 0}),
        detail::projective_reference_point(transform, {0, size, 0}),
        detail::projective_reference_point(transform, {0, 0, size})}};
}
__device__ inline float coordinate_error_bound(field_view source,
    detail::cuda_projective_pose transform, float3 point, float size)
{
    const float world_scale = fmaxf(fabsf(point.x), fmaxf(fabsf(point.y), fabsf(point.z)));
    const float pose_scale = fmaxf(fabsf(transform.translation[0]),
        fmaxf(fabsf(transform.translation[1]), fabsf(transform.translation[2])));
    return 64 * FLT_EPSILON * fmaxf(1.f,
        fmaxf((world_scale + pose_scale + 16 * size) / source.voxel_size,
            fabsf(source.origin / source.voxel_size)));
}
}
