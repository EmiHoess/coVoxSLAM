#pragma once
#include <covoxslam/sensor_grid_cuda.hpp>
#include "cli_lookup.hpp"
#include <charconv>

namespace covoxslam
{
enum class execution_choice { unknown, stream, graph };
inline constexpr cli_option_entry<execution_choice> execution_choices[]{
    {"stream", execution_choice::stream}
};
inline operation_status select_sensor_execution(std::string_view text, sensor_grid::gather_options& options)
{
    switch (lookup_cli_option(text, execution_choices)) {
    case execution_choice::stream: options.execution = sensor_grid::execution_mode::stream; break;
    default: return std::unexpected(operation_error{"coVoxSLAM requires stream sensor execution"});
    }
    return {};
}
enum class gather_choice { unknown, csr, warp, interleaved, hybrid, subwarp, short_interleaved, aligned, ray16, compact, pipelined, block_long, combined, uncapped };
inline constexpr cli_option_entry<gather_choice> gather_choices[]{
    {"hybrid-combined", gather_choice::combined}
};
inline operation_status select_sensor_gather(std::string_view text, sensor_grid::gather_options& options)
{
    switch (lookup_cli_option(text, gather_choices)) {
    case gather_choice::combined: options.backend = sensor_grid::gather_backend::hybrid_combined; break;
    default: return std::unexpected(operation_error{"coVoxSLAM requires hybrid-combined gathering"});
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
    case sensor_grid::gather_backend::hybrid_combined: return "cuda-sensor-grid-hybrid-combined";
    case sensor_grid::gather_backend::hybrid_combined_uncapped: return "cuda-sensor-grid-hybrid-combined-uncapped";
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
