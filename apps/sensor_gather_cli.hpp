#pragma once
#include <covoxslam/sensor_grid_cuda.hpp>
#include "cli_lookup.hpp"

namespace covoxslam
{
enum class gather_choice { unknown, csr, warp, interleaved };
inline constexpr cli_option_entry<gather_choice> gather_choices[]{
    {"csr", gather_choice::csr}, {"warp-soa-prepared", gather_choice::warp},
    {"thread-interleaved-prepared", gather_choice::interleaved}
};
inline operation_status select_sensor_gather(std::string_view text, sensor_grid::gather_options& options)
{
    switch (lookup_cli_option(text, gather_choices)) {
    case gather_choice::csr: options.backend = sensor_grid::gather_backend::csr; break;
    case gather_choice::warp: options.backend = sensor_grid::gather_backend::warp_soa_prepared; break;
    case gather_choice::interleaved: options.backend = sensor_grid::gather_backend::thread_interleaved_prepared; break;
    default: return std::unexpected(operation_error{"unsupported sensor gather backend"});
    }
    return {};
}
inline const char* sensor_gather_name(sensor_grid::gather_backend mode)
{
    switch (mode) {
    case sensor_grid::gather_backend::csr: return "cuda-sensor-grid";
    case sensor_grid::gather_backend::warp_soa_prepared: return "cuda-sensor-grid-warp-soa-prepared";
    case sensor_grid::gather_backend::thread_interleaved_prepared: return "cuda-sensor-grid-thread-interleaved-prepared";
    }
    return "invalid";
}
}
