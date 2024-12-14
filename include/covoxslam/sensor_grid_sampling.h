#pragma once

#include <covoxslam/sensor_grid_types.h>
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
enum class interpolation : std::uint32_t { nearest, tetrahedral, trilinear, exact_overlap };
struct sample_value { float distance{}, weight{}; bool valid{}; };
namespace sampling_detail
{
covoxslam_sample_callable inline bool finite(float value) { return value >= -FLT_MAX && value <= FLT_MAX; }
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

    if (!view.voxels || !view.side || view.side > unsigned(INT_MAX - 1) ||
        !(view.voxel_size > 0) || !finite(view.voxel_size) || !finite(view.origin) ||
        !finite(x) || !finite(y) || !finite(z)) return {};
    const std::size_t side = view.side;
    if (side > std::size_t(-1) / side / side / sizeof(voxel)) return {};
    const float u[]{center_coordinate(x, view), center_coordinate(y, view), center_coordinate(z, view)};

    for (unsigned axis = 0; axis < 3; ++axis)
        if (!finite(u[axis]) || u[axis] < -1 || !(u[axis] < float(view.side))) return {};
    accumulator result;

    if (mode == interpolation::nearest)
    {
        if (!result.add(view, int(floorf(u[0] + .5f)), int(floorf(u[1] + .5f)), int(floorf(u[2] + .5f)), 1)) return {};
        return result.finish();
    }
    int base[3]{};
     float fraction[3]{};

    for (unsigned axis = 0; axis < 3; ++axis) {
        const float lower = floorf(u[axis]);
        base[axis] = static_cast<int>(lower);
        fraction[axis] = u[axis] - lower;
    }

    if (mode == interpolation::tetrahedral)
    {
        unsigned order[]{0, 1, 2};

        for (unsigned i = 0; i < 2; ++i)
            for (unsigned j = i + 1; j < 3; ++j)
                if (fraction[order[j]] > fraction[order[i]]) {
                    const auto temporary = order[i]; order[i] = order[j]; order[j] = temporary;
                }
        const float coefficients[]{1 - fraction[order[0]], fraction[order[0]] - fraction[order[1]],
            fraction[order[1]] - fraction[order[2]], fraction[order[2]]};
        int corner[]{base[0], base[1], base[2]};

        for (unsigned vertex = 0; vertex < 4; ++vertex)
        {
            if (vertex) ++corner[order[vertex - 1]];
            if (!result.add(view, corner[0], corner[1], corner[2], coefficients[vertex])) return {};
        }
    }
    else if (mode == interpolation::trilinear)
    {
        for (unsigned corner = 0; corner < 8; ++corner) {
            const unsigned dx = corner & 1, dy = (corner >> 1) & 1, dz = corner >> 2;
            const float alpha = (dx ? fraction[0] : 1 - fraction[0]) *
                (dy ? fraction[1] : 1 - fraction[1]) * (dz ? fraction[2] : 1 - fraction[2]);
            if (!result.add(view, base[0] + int(dx), base[1] + int(dy), base[2] + int(dz), alpha)) return {};
        }
    }
    else return {};
    return result.finish();
}
}
#undef covoxslam_sample_callable
