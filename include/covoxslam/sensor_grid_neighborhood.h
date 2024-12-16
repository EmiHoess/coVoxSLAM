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
    float x, float y, float z, std::uint32_t samples);
}
#undef covoxslam_neighborhood_callable
