#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace covoxslam
{
struct tsdf_record
{
    float distance{};
    float weight{};
     std::uint32_t sample_count{};
    std::uint32_t flags{};
};

struct esdf_record {
    float signed_distance_m{};
    std::int32_t parent_dx{};
    std::int32_t parent_dy{};
    std::int32_t parent_dz{};
     std::uint32_t flags{};
}
;

}
