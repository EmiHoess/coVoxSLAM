#pragma once
#include <covoxslam/sensor_grid_sampling.h>

#ifdef __CUDACC__
#define covoxslam_neighborhood_callable __host__ __device__
#else
#define covoxslam_neighborhood_callable
#endif

namespace covoxslam::sensor_grid
{
covoxslam_neighborhood_callable inline bool valid_neighborhood_samples(std::uint32_t samples)
{
    return samples == 0 || samples == 8 || samples == 27;
}
covoxslam_neighborhood_callable inline sample_value sample_neighborhood(field_view view,
    float x, float y, float z, std::uint32_t samples)
{
    if (!samples) return sample(view, x, y, z, interpolation::nearest);
    using namespace sampling_detail;
    if (!valid_neighborhood_samples(samples) || !valid_view(view) || !finite(x) || !finite(y) || !finite(z)) return {};
    const float u[]{center_coordinate(x, view), center_coordinate(y, view), center_coordinate(z, view)};
    int base[3]{};
    const int width = samples == 8 ? 2 : 3;

    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (!finite(u[axis]) || u[axis] < -1 || !(u[axis] < float(view.side))) return {};
        base[axis] = samples == 8 ? int(floorf(u[axis])) : int(floorf(u[axis] + .5f)) - 1;
    }
    accumulator result;

    for (int dz = 0; dz < width; ++dz)
        for (int dy = 0; dy < width; ++dy)
            for (int dx = 0; dx < width; ++dx)
                if (!result.add(view, base[0] + dx, base[1] + dy, base[2] + dz, 1)) return {};
    result.numerator /= samples;
     result.denominator /= samples;
    return result.finish();
}
}
#undef covoxslam_neighborhood_callable
