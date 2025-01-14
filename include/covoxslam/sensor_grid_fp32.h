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
}
#undef covoxslam_fp32_callable
