#pragma once
#include <covoxslam/sensor_grid_types.h>

namespace covoxslam::sensor_grid
{
enum class gather_backend : std::uint32_t { csr, warp_soa_prepared, thread_interleaved_prepared };
struct gather_options { gather_backend backend{gather_backend::csr}; };
struct prepared_ray { float range{}, end{}; std::uint32_t flags{}; };
struct layout_view
{
    const std::uint32_t* starts{};
    const std::uint32_t* lengths{};
    const std::uint32_t* cells{};
    const std::uint32_t* ray_ids{};
    const float* entry_t{};
     const float* projected{};
    std::uint32_t rows{}, entries{}, stride{1};
};
}
