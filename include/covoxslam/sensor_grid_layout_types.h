#pragma once
#include <covoxslam/sensor_grid_types.h>

namespace covoxslam::sensor_grid
{
enum class gather_backend : std::uint32_t { csr, warp_soa_prepared, thread_interleaved_prepared, warp_soa_hybrid_prepared,
    warp_soa_subwarp_prepared };
struct gather_options { gather_backend backend{gather_backend::csr}; std::uint32_t short_list_max{16}; };
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
    const std::uint32_t* row_ids{};
    std::uint32_t short_rows{};
    std::uint32_t bin_ends[2][4]{};
};
}
