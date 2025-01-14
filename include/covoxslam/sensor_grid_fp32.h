#pragma once
#include <covoxslam/sensor_grid_sampling.h>

#if defined(__CUDACC__)
#define covoxslam_fp32_callable __host__ __device__
#else
#define covoxslam_fp32_callable
#endif

namespace covoxslam::sensor_grid
{
enum class transfer_arithmetic : std::uint32_t { fp64, fp32 };
namespace fp32_sampling
{
struct accumulator
{
    float numerator{}, denominator{};
    covoxslam_fp32_callable bool add(field_view view, int x, int y, int z, float alpha)
    {
        using sampling_detail::finite;
        if (!(alpha >= 0) || !finite(alpha)) return false;

        if (!(alpha > 8 * FLT_EPSILON)) return true;

        if (x < 0 || y < 0 || z < 0 || unsigned(x) >= view.side || unsigned(y) >= view.side || unsigned(z) >= view.side)
            return false;
        const auto value = view.voxels[std::size_t(x) + view.side * (std::size_t(y) + std::size_t(view.side) * z)];

        if (!value.contributions || !(value.weight > 0) || !finite(value.weight) || !finite(value.distance)) return false;
        const float weight = alpha * value.weight;
        numerator += weight * value.distance;
         denominator += weight;
        return true;
    }
};
}
covoxslam_fp32_callable inline sample_value sample_fp32(field_view view, float x, float y, float z, interpolation mode)
{
    using namespace sampling_detail;
    if (mode == interpolation::nearest) return sample(view, x, y, z, mode);

    if (!valid_view(view) || !finite(x) || !finite(y) || !finite(z)) return {};
    const float u[]{center_coordinate(x, view), center_coordinate(y, view), center_coordinate(z, view)};

    for (unsigned axis = 0; axis < 3; ++axis)
        if (!finite(u[axis]) || u[axis] < -1 || !(u[axis] < float(view.side))) return {};
    fp32_sampling::accumulator result;

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
    const float distance = result.numerator / result.denominator;
    if (!(result.denominator > 0) || !finite(result.denominator) || !finite(distance))
        return sample(view, x, y, z, mode);
    return {distance, result.denominator, true};
}
}
#undef covoxslam_fp32_callable
