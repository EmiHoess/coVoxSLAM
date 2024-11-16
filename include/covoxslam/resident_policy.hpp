#pragma once

#include <cstdint>

namespace covoxslam
{
enum class cuda_resident_order : std::uint32_t { ordered, arrival };

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
}
