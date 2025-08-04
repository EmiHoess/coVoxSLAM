#pragma once
#include <covoxslam/sensor_grid_cuda.hpp>
#include "cli_lookup.hpp"
#include <charconv>

namespace covoxslam
{
enum class execution_choice { unknown, stream };
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
enum class gather_choice { unknown, combined };
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
    case sensor_grid::gather_backend::hybrid_combined: return "cuda-sensor-grid-hybrid-combined";
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
