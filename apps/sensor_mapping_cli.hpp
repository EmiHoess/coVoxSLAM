#pragma once
#include <covoxslam/sensor_grid_sampling.h>
#include "cli_lookup.hpp"

namespace covoxslam
{
enum class sensor_backend_choice { unknown, combined };
inline constexpr cli_option_entry<sensor_backend_choice> sensor_backend_choices[]{
    {"cuda-sensor-grid-combined", sensor_backend_choice::combined}
};
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
inline const char* sensor_sampling_name(sensor_grid::interpolation mode)
{
    switch (mode) {
    case sensor_grid::interpolation::nearest: return "sensor-nearest";
    }
    return "invalid";
}
}
