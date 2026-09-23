#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/mapping/tsdf/active_submap.hpp>
#include <covoxslam/mapping/tsdf/detail/sampling.hpp>
#include "mapping_cli.hpp"
#include "sensor_gather_cli.hpp"

#include <string_view>

namespace covoxslam
{
enum class sensor_mapping_option { unknown, backend, interpolation, gather, threshold, block_threshold, execution, source_mask, transfer_precision, transfer_kernel, candidates, candidate_capacity, discovery };
inline constexpr cli_option_entry<sensor_mapping_option> sensor_mapping_options[]{
    {"--backend", sensor_mapping_option::backend}, {"--sensor-grid-interpolation", sensor_mapping_option::interpolation},
    {"--sensor-grid-gather", sensor_mapping_option::gather},
    {"--sensor-grid-short-list-max", sensor_mapping_option::threshold},
    {"--sensor-grid-block-list-min", sensor_mapping_option::block_threshold},
    {"--sensor-grid-execution", sensor_mapping_option::execution},
    {"--sensor-grid-source-mask", sensor_mapping_option::source_mask},
    {"--sensor-grid-transfer-precision", sensor_mapping_option::transfer_precision},
    {"--sensor-grid-transfer", sensor_mapping_option::transfer_kernel},
    {"--sensor-grid-candidates", sensor_mapping_option::candidates},
    {"--sensor-grid-candidate-capacity", sensor_mapping_option::candidate_capacity},
    {"--sensor-grid-discovery", sensor_mapping_option::discovery}
};
enum class discovery_choice { unknown, gpu_resident };
inline constexpr cli_option_entry<discovery_choice> discovery_choices[]{
    {"gpu-resident", discovery_choice::gpu_resident}
};
enum class candidate_choice { unknown, whole_tiles };
inline constexpr cli_option_entry<candidate_choice> candidate_choices[]{
    {"whole-tiles", candidate_choice::whole_tiles}
};
enum class source_mask_choice { unknown, dense };
inline constexpr cli_option_entry<source_mask_choice> source_mask_choices[]{
    {"dense", source_mask_choice::dense}
};
enum class transfer_precision_choice { unknown, fp64 };
enum class transfer_kernel_choice { unknown, reference };
inline constexpr cli_option_entry<transfer_kernel_choice> transfer_kernel_choices[]{
    {"reference", transfer_kernel_choice::reference}
};
inline constexpr cli_option_entry<transfer_precision_choice> transfer_precision_choices[]{
    {"64", transfer_precision_choice::fp64}
};
enum class sensor_sampling_value { unknown, nearest };
inline constexpr cli_option_entry<sensor_sampling_value> sensor_sampling_values[]{
    {"nearest", sensor_sampling_value::nearest}
};
inline operation_result<bool> parse_sensor_mapping_option(std::string_view key, std::string_view value,
    bool& sensor_selected, bool& mapping_enabled, sensor_grid::gather_options& gather,
    sensor_grid::transfer_options& transfer, sensor_grid::interpolation& sampling)
{
    switch (lookup_cli_option(key, sensor_mapping_options)) {
    case sensor_mapping_option::backend:
        switch (lookup_cli_option(value, mapping_backend_choices)) {
        case mapping_backend_choice::combined:
            sensor_selected = mapping_enabled = true;
            gather.backend = sensor_grid::gather_backend::hybrid_combined;
            return true;
        default: break;
        }
        sensor_selected = false;
        return false;
    case sensor_mapping_option::interpolation:
        switch (lookup_cli_option(value, sensor_sampling_values)) {
        case sensor_sampling_value::nearest: sampling = sensor_grid::interpolation::nearest; break;
        default: return std::unexpected(operation_error{"unsupported sensor interpolation"});
        }
        return true;
    case sensor_mapping_option::threshold: {
        const auto selected = select_sensor_threshold(value, gather);

        if (!selected) return std::unexpected(selected.error());
        return true;
    }
    case sensor_mapping_option::block_threshold: {
        const auto selected = select_block_threshold(value, gather);
        if (!selected) return std::unexpected(selected.error());
        return true;
    }
    case sensor_mapping_option::gather: {
        const auto selected = select_sensor_gather(value, gather);

        if (!selected) return std::unexpected(selected.error());
        return true;
    }
    case sensor_mapping_option::execution: {
        const auto selected = select_sensor_execution(value, gather);
        if (!selected) return std::unexpected(selected.error());
        return true;
    }
    case sensor_mapping_option::source_mask:
        switch (lookup_cli_option(value, source_mask_choices)) {
        case source_mask_choice::dense: transfer.source_mask = sensor_grid::source_mask_mode::dense; break;
        default: return std::unexpected(operation_error{"coVoxSLAM requires a dense sensor source mask"});
        }
        return true;
    case sensor_mapping_option::transfer_precision:
        switch (lookup_cli_option(value, transfer_precision_choices)) {
        case transfer_precision_choice::fp64: transfer.arithmetic = sensor_grid::transfer_arithmetic::fp64; break;
        default: return std::unexpected(operation_error{"coVoxSLAM requires 64 bit sensor transfer"});
        }
        return true;
    case sensor_mapping_option::transfer_kernel:
        switch (lookup_cli_option(value, transfer_kernel_choices)) {
        case transfer_kernel_choice::reference: transfer.specialized = false; break;
        default: return std::unexpected(operation_error{"coVoxSLAM requires reference sensor transfer"});
        }
        return true;
    case sensor_mapping_option::candidates:
        switch (lookup_cli_option(value, candidate_choices)) {
        case candidate_choice::whole_tiles: transfer.candidates = sensor_grid::candidate_mode::whole_tiles; break;
        default: return std::unexpected(operation_error{"invalid sensor candidate method"});
        }
        return true;
    case sensor_mapping_option::discovery:
        switch (lookup_cli_option(value, discovery_choices)) {
        case discovery_choice::gpu_resident: transfer.discovery = sensor_grid::discovery_mode::gpu_resident; break;
        default: return std::unexpected(operation_error{"invalid sensor discovery method"});
        }
        return true;
    default: return false;
    }
}
inline const char* sensor_sampling_name(sensor_grid::interpolation mode)
{
    switch (mode) {
    case sensor_grid::interpolation::nearest: return "sensor-nearest";
    }
    return "invalid";
}
}
