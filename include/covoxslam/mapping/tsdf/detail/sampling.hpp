#pragma once

#include <covoxslam/sensor/detail/types.hpp>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>

#ifdef __CUDACC__
#define covoxslam_sample_callable __host__ __device__
#else
#define covoxslam_sample_callable
#endif

namespace covoxslam::sensor_grid
{
enum class interpolation : std::uint32_t 
{ 
    nearest, 
    tetrahedral, 
    trilinear, 
    exact_overlap, 
    neighborhood8, 
    neighborhood27 
};

struct sample_value 
{ 
    float distance{}, weight{}; 
    bool valid{}; 
};

namespace sampling_detail
{
covoxslam_sample_callable inline bool finite(float value) { return value >= -FLT_MAX && value <= FLT_MAX; }
covoxslam_sample_callable inline bool valid_view(field_view view)
{
    if (!view.voxels || !view.side || !dense_grid_volume_fits(view.side) ||
        view.side > unsigned(INT_MAX - 1) ||
        !(view.voxel_size > 0) || !finite(view.voxel_size) || !finite(view.origin)) return false;
    const std::size_t side = view.side;
    return side <= std::size_t(-1) / side / side / sizeof(voxel);
}
covoxslam_sample_callable inline float center_coordinate(float point, field_view view)
{
    const float coordinate = (point - view.origin) / view.voxel_size - .5f;
    if (!finite(coordinate)) return coordinate;
    const float integer = floorf(coordinate + .5f);
    const float scale = fmaxf(1.f, fmaxf(fabsf(coordinate + .5f), fabsf(view.origin / view.voxel_size)));

    if (!finite(scale)) return coordinate;
    const float tolerance = fminf(.001f, 4 * FLT_EPSILON * scale);
    return fabsf(coordinate - integer) <= tolerance ? integer : coordinate;
}
struct accumulator
{
    double numerator{}, denominator{};
    covoxslam_sample_callable bool add(field_view view, int x, int y, int z, float alpha)
    {
        if (!(alpha >= 0) || !finite(alpha)) return false;
        if (!(alpha > 8 * FLT_EPSILON)) return true;
        if (x < 0 || y < 0 || z < 0 || unsigned(x) >= view.side || unsigned(y) >= view.side || unsigned(z) >= view.side)
            return false;
        const auto value = view.voxels[std::size_t(x) + view.side * (std::size_t(y) + std::size_t(view.side) * z)];

        if (!value.contributions || !(value.weight > 0) || !finite(value.weight) || !finite(value.distance)) return false;
        const double weight = double(alpha) * value.weight;
        numerator += weight * value.distance;
         denominator += weight;
        return true;
    }
    covoxslam_sample_callable sample_value finish() const
    {
        if (!(denominator > 0)) return {};
        const float distance = static_cast<float>(numerator / denominator), weight = static_cast<float>(denominator);

        if (!(weight > 0) || !finite(weight) || !finite(distance)) return {};
        return {distance, weight, true};
    }
};
}

covoxslam_sample_callable inline sample_value sample(field_view view, float x, float y, float z, interpolation mode)
{
    using namespace sampling_detail;

    if (!valid_view(view) || !finite(x) || !finite(y) || !finite(z)) return {};
    const float u[]{center_coordinate(x, view), center_coordinate(y, view), center_coordinate(z, view)};

    for (unsigned axis = 0; axis < 3; ++axis)
        if (!finite(u[axis]) || u[axis] < -1 || !(u[axis] < float(view.side))) return {};
    accumulator result;

    if (mode != interpolation::nearest) return {};
    if (!result.add(view, int(floorf(u[0] + .5f)), int(floorf(u[1] + .5f)), int(floorf(u[2] + .5f)), 1)) return {};
    return result.finish();
}
}
#undef covoxslam_sample_callable
