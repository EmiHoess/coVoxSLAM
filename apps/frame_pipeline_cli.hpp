#pragma once

#include <covoxslam/mapping.hpp>
#include "mapping_cli.hpp"
#include "map_export_cli.hpp"
#include "sensor_mapping_cli.hpp"
#include "pgo_cli.hpp"
#include "store_cli.hpp"
#include "retired_options.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace covoxslam
{

enum class submap_cli_option { unknown, dataset, output, frames, submaps, rings, columns, duration, verification, pose_source, preset };
inline constexpr cli_option_entry<submap_cli_option> submap_cli_options[]{
    {"--dataset", submap_cli_option::dataset}, {"--output", submap_cli_option::output},
    {"--maximum-frames", submap_cli_option::frames}, {"--maximum-submaps", submap_cli_option::submaps},
     {"--ring-stride", submap_cli_option::rings}, {"--column-stride", submap_cli_option::columns},
    {"--submap-duration-ns", submap_cli_option::duration}, {"--verification", submap_cli_option::verification},
    {"--pose-source", submap_cli_option::pose_source}, {"--preset", submap_cli_option::preset}
};
enum class verification_option { unknown, cached, full };
inline constexpr cli_option_entry<verification_option> verification_options[]{
    {"cached", verification_option::cached}, {"full", verification_option::full}
};
enum class trajectory_option { unknown, odometry, ground_truth };
inline constexpr cli_option_entry<trajectory_option> trajectory_options[]{
    {"odometry", trajectory_option::odometry}, {"ground_truth", trajectory_option::ground_truth}
};

inline int run_submap_cli(int argc, char** argv)
{
        const auto report_error = [](std::string_view message) {
            std::cerr << "covoxslam: " << message << '\n';
            return 2;
        };
        pipeline_options options;
        mapping_config mapping_options;
        bool mapping_enabled{};
        bool sensor_selected{};
        bool sensor_options_set{};
        sensor_grid::gather_options sensor_gather;
        sensor_grid::transfer_options sensor_transfer;
        auto sensor_mode = sensor_grid::interpolation::nearest;
        mapping_device_options cuda_options;
        dataset_load_options input_options;
        std::filesystem::path root;
        std::filesystem::path output;
        bool preset_specified{};
        unsigned selectors_seen{};
        pgo_cli_options pgo_settings;
        store_pipeline_options store_settings;

        auto preset = sensor_grid::combined_nearest_preset();

        if (!preset) return report_error(preset.error().message);
        mapping_options = preset->mapping;
        mapping_options.esdf_generator = esdf_backend::cuda_block_band_soa;
         sensor_gather = preset->gather;
        sensor_transfer = preset->transfer;
        sensor_mode = preset->sampling;
        options.ring_stride = preset->ring_stride;
        options.column_stride = preset->column_stride;
        sensor_selected = mapping_enabled = true;

        for (int index = 2; index < argc; index += 2) {
            if (index + 1 >= argc) return report_error("missing submap option value");
            unsigned selector{};
            std::string_view required;
            switch (lookup_cli_option(argv[index], store_options)) {
            case store_option::mapping: selector = 1; required = "resident-soa"; break;
            case store_option::layout: selector = 2; required = "soa"; break;
            default: break;
            }

            if (lookup_cli_option(argv[index], covoxslam::mapping_options) == mapping_option::esdf_generator) {
                selector = 4;
                 required = "cuda-block-band-soa";
            }
            if (selector)
            {
                if (selectors_seen & selector) return report_error("repeated mapping selector");
                if (std::string_view(argv[index + 1]) != required)
                    return report_error("mapping selector is not applicable to coVoxSLAM");
                selectors_seen |= selector;
            }
            if (lookup_cli_option(argv[index], submap_cli_options) != submap_cli_option::preset) continue;

            if (preset_specified || std::string_view(argv[index + 1]) != "combined-nearest")
                return report_error("expected one combined-nearest preset");
            preset_specified = true;
        }
        for (int index = 2; index < argc; index += 2)
        {
            const std::string_view option(argv[index]);
            if (index + 1 >= argc) return report_error("missing submap option value");
            const std::string_view value(argv[index + 1]);
            const auto active = validate_active_option(option);

            if (!active) return report_error(active.error().message);
            const auto store_parsed = parse_store_option(option, value, store_settings);
            if (!store_parsed) return report_error(store_parsed.error().message);
            if (*store_parsed) continue;
            const auto pgo_parsed = parse_pgo_option(option, value, pgo_settings);
            if (!pgo_parsed) return report_error(pgo_parsed.error().message);
            if (*pgo_parsed) continue;
            switch (lookup_cli_option(option, sensor_mapping_options)) {
            case sensor_mapping_option::backend:
                switch (lookup_cli_option(value, sensor_backend_choices)) {
                case sensor_backend_choice::combined:
                    sensor_selected = mapping_enabled = true;
                    sensor_gather.backend = sensor_grid::gather_backend::hybrid_combined;
                    continue;
                default: break;
                }
                sensor_selected = false;
                break;
            case sensor_mapping_option::interpolation:
                sensor_options_set = true;
                switch (lookup_cli_option(value, sensor_sampling_values)) {
                case sensor_sampling_value::nearest: sensor_mode = sensor_grid::interpolation::nearest; break;
                default: return report_error("unsupported sensor interpolation");
                }
                continue;
            case sensor_mapping_option::threshold: {
                const auto selected = select_sensor_threshold(value, sensor_gather);
                if (!selected) return report_error(selected.error().message);
                sensor_options_set = true;
                continue;
            }
            case sensor_mapping_option::block_threshold: {
                const auto selected = select_block_threshold(value, sensor_gather);
                if (!selected) return report_error(selected.error().message);
                sensor_options_set = true;
                continue;
            }
            case sensor_mapping_option::gather: {
                const auto selected = select_sensor_gather(value, sensor_gather);
                if (!selected) return report_error(selected.error().message);
                sensor_options_set = true;
                continue;
            }
            case sensor_mapping_option::execution: {
                const auto selected = select_sensor_execution(value, sensor_gather);
                if (!selected) return report_error(selected.error().message);
                sensor_options_set = true;
                continue;
            }
            case sensor_mapping_option::source_mask:
                switch (lookup_cli_option(value, source_mask_choices)) {
                case source_mask_choice::dense: sensor_transfer.source_mask = sensor_grid::source_mask_mode::dense; break;
                default: return report_error("coVoxSLAM requires a dense sensor source mask");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::transfer_precision:
                switch (lookup_cli_option(value, transfer_precision_choices)) {
                case transfer_precision_choice::fp64: sensor_transfer.arithmetic = sensor_grid::transfer_arithmetic::fp64; break;
                default: return report_error("coVoxSLAM requires 64 bit sensor transfer");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::transfer_kernel:
                switch (lookup_cli_option(value, transfer_kernel_choices)) {
                case transfer_kernel_choice::reference: sensor_transfer.specialized = false; break;
                default: return report_error("coVoxSLAM requires reference sensor transfer");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::candidates:
                switch (lookup_cli_option(value, candidate_choices)) {
                case candidate_choice::whole_tiles: sensor_transfer.candidates = sensor_grid::candidate_mode::whole_tiles; break;
                default: return report_error("invalid sensor candidate method");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::discovery:
                switch (lookup_cli_option(value, discovery_choices)) {
                case discovery_choice::gpu_resident: sensor_transfer.discovery = sensor_grid::discovery_mode::gpu_resident; break;
                default: return report_error("invalid sensor discovery method");
                }
                sensor_options_set = true;
                continue;
            default: break;
            }
            switch (lookup_cli_option(option, submap_cli_options))
            {
            case submap_cli_option::preset: continue;
            case submap_cli_option::dataset: root = argv[index + 1]; continue;
            case submap_cli_option::pose_source:
                switch (lookup_cli_option(value, trajectory_options))
                {
                case trajectory_option::odometry: options.pose_source = pose_source::odometry; break;
                 case trajectory_option::ground_truth: options.pose_source = pose_source::ground_truth; break;
                default: return report_error("unsupported trajectory source");
                }
                continue;
            case submap_cli_option::verification:
                switch (lookup_cli_option(value, verification_options)) {
                case verification_option::cached: input_options.verification = dataset_verification_mode::prefer_cache; break;
                 case verification_option::full: input_options.verification = dataset_verification_mode::force_full; break;
                default: return report_error("unsupported input verification mode");
                }
                continue;
            case submap_cli_option::duration:
            {
                const auto converted = std::from_chars(value.data(), value.data() + value.size(), options.submap_duration_ns);

                if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size())
                    return report_error("expected an unsigned duration in nanoseconds");
                continue;
            }
            case submap_cli_option::output:
                if (value.empty()) return report_error("--output requires a directory");
                output = argv[index + 1]; continue;
            case submap_cli_option::frames:
            case submap_cli_option::submaps:
            case submap_cli_option::rings:
            case submap_cli_option::columns:
            {
                std::size_t maximum{};
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), maximum);

                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
                    return report_error("expected an addressable unsigned count");
                switch (lookup_cli_option(option, submap_cli_options)) {
                case submap_cli_option::frames: options.maximum_frames = maximum; break;
                case submap_cli_option::submaps: options.maximum_submaps = maximum; break;
                 case submap_cli_option::rings: options.ring_stride = maximum; break;
                case submap_cli_option::columns: options.column_stride = maximum; break;
                default: break;
                }
                continue;
            }
            default: break;
            }
            {
                const auto cuda_parsed = parse_mapping_device(option, value, mapping_options,
                    cuda_options.device, cuda_options.maximum_device_bytes);

                if (!cuda_parsed) return report_error(cuda_parsed.error().message);
                if (*cuda_parsed) continue;
                const auto parsed = parse_mapping_option(option, value, mapping_options, mapping_enabled);

                if (!parsed) return report_error(parsed.error().message);
                if (!*parsed) return report_error("unsupported submap option");
            }
        }
         if (root.empty()) return report_error("--submaps requires --dataset directory");
        if (!sensor_selected || !sensor_grid::matches_combined_nearest({mapping_options, sensor_gather,
                sensor_transfer, sensor_mode, options.ring_stride, options.column_stride}))
            return report_error("explicit options conflict with combined-nearest preset");
        if (sensor_options_set && !sensor_selected) return report_error("sensor options require cuda-sensor-grid");
        if (!options.submap_duration_ns || !options.ring_stride || !options.column_stride)
            return report_error("duration and sampling strides must be positive");
        pgo_settings.solver.device = cuda_options.device;
        store_settings.device = cuda_options.device;
        store_settings.esdf_generator = mapping_options.esdf_generator;
        auto store_valid = validate_store_pipeline(store_settings, pgo_settings.solver);
        if (!store_valid) return report_error(store_valid.error().message);
        if (store_settings.resident_mapping && (!sensor_selected ||
            !sensor_grid::combined_gather(sensor_gather.backend) || sensor_mode != sensor_grid::interpolation::nearest ||
            sensor_transfer.discovery != sensor_grid::discovery_mode::gpu_resident))
            return report_error("resident mapping storage requires combined nearest with resident discovery");
        if (store_settings.enabled && !mapping_enabled) return report_error("map storage requires mapping");
        if (!store_settings.checkpoint_root.empty() && !output.empty()) {
            std::error_code error;
            auto store_path = std::filesystem::weakly_canonical(store_settings.checkpoint_root, error);
            if (error) return report_error("checkpoint path is invalid");
            auto export_path = std::filesystem::weakly_canonical(output, error);
            if (error) return report_error("export path is invalid");
            if (store_path == export_path) return report_error("checkpoint and export paths must differ");
        }
        const auto loaded = load_dataset(root, input_options);
        if (pgo_settings.publication_selected && !pgo_settings.solver.enabled)
            return report_error("PGO publication requires an enabled solver");
        if (pgo_settings.solver.enabled && !mapping_enabled)
            return report_error("PGO requires map construction");
        const auto pgo_valid = pgo::validate_options(pgo_settings.solver);
        if (!pgo_valid) return report_error(pgo_valid.error().message);
        if (!loaded) return report_error(loaded.error().message);
        const auto& input = *loaded;
        auto mapper_options = mapping_options;
        std::optional<operation_error> resident_failure;
        const std::string backend_label = "coVoxSLAM";
        std::cout << "coVoxSLAM, nearest cell TSDF, resident SoA block band ESDF\n";
        const auto fusion_label = sensor_sampling_name(sensor_mode);
        std::unique_ptr<map_exporter> exporter;
        std::vector<submap> pgo_maps;
        std::unique_ptr<resident_pipeline> resident_maps;
        if (store_settings.resident_mapping) {
            auto created = resident_pipeline::create(store_settings, pgo_settings.solver);
            if (!created) return report_error(created.error().message);
            resident_maps = std::move(*created);
            std::cout << "mapping storage resident-soa, device handoff, deferred canonical readback\n";
        }
        const auto publish = [&](const submap& completed) { pgo_maps.push_back(completed); };
        auto selected = sensor_mapping_callbacks(input.sensor, mapper_options, cuda_options.device,
            resident_failure, publish, {}, sensor_mode, sensor_gather, cuda_options.maximum_device_bytes,
            sensor_transfer, resident_maps.get());

        if (!selected) return report_error(selected.error().message);
        auto callbacks = std::move(*selected);
        std::cout << "sensor field uses calibrated ranges and scan start poses\n";
        const auto result = run_frame_pipeline(input, options, callbacks);

        if (result.cancelled) return report_error(resident_failure ? resident_failure->message : "mapping cancelled");
        if (pgo_settings.solver.enabled || store_settings.enabled) {
            auto optimized = [&]() -> operation_result<pgo_cli_result> {
                auto stored = resident_maps->finish(pgo_maps, pgo_settings.solver, pgo_settings.shadow);
                if (!stored) return std::unexpected(operation_error{
                    "stored pipeline stopped before export: " + stored.error().message});
                auto ready = validate_stored_export(pgo_maps, store_settings);
                if (!ready) return std::unexpected(ready.error());
                return pgo_cli_result{stored->backend, stored->hash, stored->publication, stored->report};
            }();
            if (!optimized) return report_error(optimized.error().message);
            std::cout << optimized->report << "pose publication " << optimized->publication << '\n';
            exporter = open_map_export(output, input, mapping_options, backend_label.c_str(), fusion_label,
                optimized->backend, optimized->hash, optimized->publication);
            if (exporter) {
                std::ofstream diagnostics(output / "pgo.txt", std::ios::trunc);
                diagnostics << optimized->report << "pose publication " << optimized->publication << '\n';
                diagnostics.close();
                if (!diagnostics) return report_error("failed to write solver diagnostics");

                for (const auto& map : pgo_maps) exporter->write(map);
            }
        }
        if (exporter) exporter->finish();
        std::cout << std::setprecision(17);
        for (const auto& submap : result.submaps) {
            const auto& anchor = submap.anchor;
            const auto& initial = submap.initial_pose;
             std::cout << "submap " << submap.id << ": " << submap.frame_ids.size()
                      << " frames, " << submap.start_stamp_ns << ".." << submap.end_stamp_ns
                      << " ns, anchor " << anchor.x << ' ' << anchor.y << ' ' << anchor.z
                      << ' ' << anchor.yaw << ", initial translation " << initial.translation.x
                      << ' ' << initial.translation.y << ' ' << initial.translation.z
                      << ", initial quaternion " << initial.rotation.x << ' ' << initial.rotation.y
                      << ' ' << initial.rotation.z << ' ' << initial.rotation.w << '\n';
        }
        std::cout << result.frames_loaded << " frames, " << result.submaps.size()
                   << " submaps, " << result.rays_generated << " rays\n";
        const auto& timing = result.timing;
        std::cout << "payload bytes " << timing.payload_bytes << ", read ms " << timing.read_ms
            << ", validation ms " << timing.validation_ms << ", decode ms " << timing.decode_ms
            << ", deskew ms " << timing.deskew_ms << ", integration ms " << timing.integration_ms
            << ", buffer to map ms " << timing.buffer_to_map_ms()
            << ", finalization ms " << timing.finalization_ms << '\n';
        return 0;
}

}
