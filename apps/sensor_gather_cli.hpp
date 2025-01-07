#pragma once
#include <covoxslam/sensor_grid_cuda.hpp>
#include "cli_lookup.hpp"
#include <charconv>

namespace covoxslam
{
enum class execution_choice { unknown, stream, graph };
inline constexpr cli_option_entry<execution_choice> execution_choices[]{
    {"stream", execution_choice::stream}, {"cuda-graph", execution_choice::graph}
};
inline operation_status select_sensor_execution(std::string_view text, sensor_grid::gather_options& options)
{
    switch (lookup_cli_option(text, execution_choices)) {
    case execution_choice::stream: options.execution = sensor_grid::execution_mode::stream; break;
    case execution_choice::graph: options.execution = sensor_grid::execution_mode::cuda_graph; break;
    default: return std::unexpected(operation_error{"sensor execution must be stream or cuda-graph"});
    }
    return {};
}
enum class gather_choice { unknown, csr, warp, interleaved, hybrid, subwarp, short_interleaved, aligned, ray16, compact, pipelined, block_long };
inline constexpr cli_option_entry<gather_choice> gather_choices[]{
    {"csr", gather_choice::csr}, {"warp-soa-prepared", gather_choice::warp},
    {"thread-interleaved-prepared", gather_choice::interleaved},
    {"warp-soa-hybrid-prepared", gather_choice::hybrid},
    {"warp-soa-subwarp-prepared", gather_choice::subwarp},
    {"hybrid-short-interleaved", gather_choice::short_interleaved},
    {"hybrid-aligned", gather_choice::aligned}, {"hybrid-ray16", gather_choice::ray16},
    {"hybrid-compact", gather_choice::compact}, {"hybrid-pipelined", gather_choice::pipelined},
    {"hybrid-block-long", gather_choice::block_long}
};
inline operation_status select_sensor_gather(std::string_view text, sensor_grid::gather_options& options)
{
    switch (lookup_cli_option(text, gather_choices)) {
    case gather_choice::csr: options.backend = sensor_grid::gather_backend::csr; break;
    case gather_choice::warp: options.backend = sensor_grid::gather_backend::warp_soa_prepared; break;
    case gather_choice::interleaved: options.backend = sensor_grid::gather_backend::thread_interleaved_prepared; break;
    case gather_choice::hybrid: options.backend = sensor_grid::gather_backend::warp_soa_hybrid_prepared; break;
    case gather_choice::subwarp: options.backend = sensor_grid::gather_backend::warp_soa_subwarp_prepared; break;
    case gather_choice::short_interleaved: options.backend = sensor_grid::gather_backend::hybrid_short_interleaved; break;
    case gather_choice::aligned: options.backend = sensor_grid::gather_backend::hybrid_aligned; break;
    case gather_choice::ray16: options.backend = sensor_grid::gather_backend::hybrid_ray16; break;
    case gather_choice::compact: options.backend = sensor_grid::gather_backend::hybrid_compact; break;
    case gather_choice::pipelined: options.backend = sensor_grid::gather_backend::hybrid_pipelined; break;
    case gather_choice::block_long: options.backend = sensor_grid::gather_backend::hybrid_block_long; break;
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
    case sensor_grid::gather_backend::warp_soa_hybrid_prepared: return "cuda-sensor-grid-warp-soa-hybrid-prepared";
    case sensor_grid::gather_backend::warp_soa_subwarp_prepared: return "cuda-sensor-grid-warp-soa-subwarp-prepared";
    case sensor_grid::gather_backend::hybrid_short_interleaved: return "cuda-sensor-grid-hybrid-short-interleaved";
    case sensor_grid::gather_backend::hybrid_aligned: return "cuda-sensor-grid-hybrid-aligned";
    case sensor_grid::gather_backend::hybrid_ray16: return "cuda-sensor-grid-hybrid-ray16";
    case sensor_grid::gather_backend::hybrid_compact: return "cuda-sensor-grid-hybrid-compact";
    case sensor_grid::gather_backend::hybrid_pipelined: return "cuda-sensor-grid-hybrid-pipelined";
    case sensor_grid::gather_backend::hybrid_block_long: return "cuda-sensor-grid-hybrid-block-long";
    }
    return "invalid";
}
inline operation_status select_sensor_threshold(std::string_view text, sensor_grid::gather_options& options)
{
    std::uint32_t threshold{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), threshold);

    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !threshold)
        return std::unexpected(operation_error{"sensor row threshold must be a positive integer"});
    options.short_list_max = threshold;
     return {};
}
inline operation_status select_block_threshold(std::string_view text, sensor_grid::gather_options& options)
{
    auto parsed = options;
    const auto status = select_sensor_threshold(text, parsed);
    if (status) options.block_list_min = parsed.short_list_max;
    return status;
}
}
