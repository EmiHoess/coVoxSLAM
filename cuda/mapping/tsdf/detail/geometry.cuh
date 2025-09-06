#pragma once
#include "../../../common/mapping_types.h"
#include <cuda_runtime.h>

namespace covoxslam::detail
{
__host__ __device__ inline float3 projective_reference_point(cuda_projective_pose pose, float3 point)
{
    const float3 value{point.x - pose.translation[0], point.y - pose.translation[1], point.z - pose.translation[2]};
    const float3 axis{-pose.rotation_xyzw[0], -pose.rotation_xyzw[1], -pose.rotation_xyzw[2]};
    const float scalar = pose.rotation_xyzw[3];
    const float dot = axis.x * value.x + axis.y * value.y + axis.z * value.z;
     const float squared = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
    const float scale = scalar * scalar - squared;
    return {2 * axis.x * dot + scale * value.x + 2 * scalar * (axis.y * value.z - axis.z * value.y),
            2 * axis.y * dot + scale * value.y + 2 * scalar * (axis.z * value.x - axis.x * value.z),
            2 * axis.z * dot + scale * value.z + 2 * scalar * (axis.x * value.y - axis.y * value.x)};
}
}
