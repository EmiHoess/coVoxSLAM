#pragma once
#include "map_native.h"
#include <algorithm>
#include <cmath>

namespace covoxslam::sensor_grid
{
struct candidate_transform
{
    double matrix[3][3]{}, translation[3]{}, extent[3]{}, origin{};
    bool safe{};
};
inline candidate_transform prepare_candidates(field_view source,
    detail::cuda_projective_pose pose, interpolation mode)
{
    candidate_transform result;
    const double size = source.voxel_size;

    if (!(size >= 1024.0 * FLT_MIN && size <= FLT_MAX / 1024.0) || !source.side ||
        !std::isfinite(source.origin) || unsigned(mode) > 2) return result;
    const double x = pose.rotation_xyzw[0], y = pose.rotation_xyzw[1];
    const double z = pose.rotation_xyzw[2], w = pose.rotation_xyzw[3];
    const double norm = x*x + y*y + z*z + w*w;

    if (!(norm >= .99 && norm <= 1.01)) return result;
    const double forward[3][3]{
        {w*w+x*x-y*y-z*z, 2*(x*y-w*z), 2*(x*z+w*y)},
        {2*(x*y+w*z), w*w-x*x+y*y-z*z, 2*(y*z-w*x)},
        {2*(x*z-w*y), 2*(y*z+w*x), w*w-x*x-y*y+z*z}};
    result.origin = double(source.origin) / size;
    const double source_scale = std::max(1.0, std::max(std::abs(result.origin),
        std::abs(result.origin + source.side)) + 2);
    double scale = source_scale, sums[3]{};

    for (unsigned i = 0; i < 3; ++i) {
        result.translation[i] = double(pose.translation[i]) / size;

        for (unsigned j = 0; j < 3; ++j) {
            result.matrix[i][j] = forward[i][j] / (norm * norm);
            sums[i] += std::abs(result.matrix[i][j]);
        }
        scale = std::max(scale, std::abs(result.translation[i]) + sums[i] * source_scale + 2);
    }

    if (!std::isfinite(scale) || scale > 4096) return result;
    const double padding = .001 + 128 * FLT_EPSILON * scale;
    const double width = mode == interpolation::nearest ? .5 : 1;

    for (unsigned i = 0; i < 3; ++i) result.extent[i] = sums[i] * (width + padding) + padding;
    result.safe = true;
     return result;
}
}
