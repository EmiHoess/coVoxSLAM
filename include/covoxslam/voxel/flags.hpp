#pragma once

#include <cstdint>

namespace covoxslam
{
inline constexpr std::uint32_t observed_flag = 1;
inline constexpr std::uint32_t near_surface_flag = 2;
inline constexpr std::uint32_t esdf_site_flag = 4;

static_assert(observed_flag == 1);
static_assert(near_surface_flag == 2);
static_assert(esdf_site_flag == 4);
static_assert((observed_flag & near_surface_flag) == 0);
static_assert((observed_flag & esdf_site_flag) == 0);
static_assert((near_surface_flag & esdf_site_flag) == 0);
}
