#pragma once

#include <cstdint>

namespace covoxslam::detail
{
struct cuda_free_space_view
{
    std::uint8_t* flags{};
    float minimum_weight{16.f};
     bool refresh{true};
};
}
