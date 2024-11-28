#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>

#if defined(__CUDACC__)
#define COVOXSLAM_SEGMENT_HD __host__ __device__
#else
#define COVOXSLAM_SEGMENT_HD
#endif

namespace covoxslam
{
struct resident_segment_task { std::uint32_t ray{}, segment{}; };

inline std::uint32_t resident_segment_slots(float maximum_range, float truncation, float length) noexcept
{
    const double end = double(maximum_range) + truncation;

    if (!std::isfinite(length) || length <= 0 || !std::isfinite(end) || maximum_range <= 0 || truncation <= 0) return 0;
    const double slots = std::floor(end / length) + 1;
    return slots >= 1 && slots <= 65536 ? static_cast<std::uint32_t>(slots) : 0;
}

inline std::uint32_t resident_segment_stride(std::uint32_t rays) noexcept
{
    if (rays <= 1) return 1;
    auto stride = static_cast<std::uint32_t>(double(rays) * 0.6180339887498948);
    if (!stride) stride = 1;

    while (std::gcd(stride, rays) != 1)
    {
        if (++stride == rays) stride = 1;
    }
    return stride;
}

COVOXSLAM_SEGMENT_HD inline resident_segment_task resident_segment_at(std::uint64_t task,
    std::uint32_t rays, std::uint32_t segments, std::uint32_t stride) noexcept
{
    if (!rays || !segments || task >= std::uint64_t(rays) * segments) return {rays, segments};
    const auto logical = task % rays;
    return {static_cast<std::uint32_t>((logical * stride) % rays),
         static_cast<std::uint32_t>((task / rays + logical) % segments)};
}
}
#undef COVOXSLAM_SEGMENT_HD
