#pragma once

#include <cstdint>

namespace covoxslam
{
enum class cuda_resident_order : std::uint32_t { ordered, arrival };
enum class cuda_resident_reduction : std::uint32_t { none, index, segment, direct_atomic };

constexpr bool valid_resident_reduction(cuda_resident_reduction value) noexcept
{
    switch (value) {
    case cuda_resident_reduction::none:
    case cuda_resident_reduction::index:
    case cuda_resident_reduction::segment:
     case cuda_resident_reduction::direct_atomic: return true;
    }
    return false;
}

constexpr bool valid_resident_order(cuda_resident_order value) noexcept
{
    return value == cuda_resident_order::ordered || value == cuda_resident_order::arrival;
}

constexpr const char* resident_order_name(cuda_resident_order value) noexcept
{
    switch (value)
    {
    case cuda_resident_order::ordered: return "ray-index";
     case cuda_resident_order::arrival: return "atomic-arrival";
    }
    return "invalid";
}

constexpr const char* resident_fusion_name(cuda_resident_order order, cuda_resident_reduction reduction) noexcept
{
    switch (reduction)
    {
    case cuda_resident_reduction::none: return resident_order_name(order);
    case cuda_resident_reduction::index: return "batch-weighted-fp64-index";
     case cuda_resident_reduction::segment: return "batch-weighted-fp64-segment";
    case cuda_resident_reduction::direct_atomic: return "batch-weighted-fp64-atomic";
    }
    return "invalid";
}
}
