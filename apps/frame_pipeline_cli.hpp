#pragma once

#include <covoxslam/mapping.hpp>
#include "projective_policy_cli.hpp"
#include "ray_fusion_cli.hpp"
#include "cuda_esdf_cli.hpp"
#include "map_export_cli.hpp"
#include "resident_cli.hpp"
#include "sensor_mapping_cli.hpp"
#include "pgo_cli.hpp"
#include "store_cli.hpp"

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
        auto sensor_mode = sensor_grid::interpolation::trilinear;
        cuda_mapping_cli_options cuda_options;
        resident_cli_options resident_options;
        ray_fusion_cli_options fusion_options;
        cuda_projective_options projective_options;
        bool projective_option_set{};
        bool projective_policy_option_set{};
         bool projective_control_set{};
        cuda_blocks_cli_options esdf_options;
        dataset_load_options input_options;
        std::filesystem::path root;
        std::filesystem::path output;
        bool nearest_preset{true};
        bool preset_specified{};
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
            const auto store_parsed = parse_store_option(option, value, store_settings);
            if (!store_parsed) return report_error(store_parsed.error().message);
            if (*store_parsed) continue;
            const auto pgo_parsed = parse_pgo_option(option, value, pgo_settings);
            if (!pgo_parsed) return report_error(pgo_parsed.error().message);
            if (*pgo_parsed) continue;
            switch (lookup_cli_option(option, sensor_mapping_options)) {
            case sensor_mapping_option::backend:
                switch (lookup_cli_option(value, sensor_backend_choices)) {
                case sensor_backend_choice::sensor:
                    sensor_selected = mapping_enabled = true;
                    continue;
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
                case sensor_sampling_value::tetrahedral: sensor_mode = sensor_grid::interpolation::tetrahedral; break;
                case sensor_sampling_value::trilinear: sensor_mode = sensor_grid::interpolation::trilinear; break;
                case sensor_sampling_value::overlap: sensor_mode = sensor_grid::interpolation::exact_overlap; break;
                case sensor_sampling_value::neighborhood8: sensor_mode = sensor_grid::interpolation::neighborhood8; break;
                case sensor_sampling_value::neighborhood27: sensor_mode = sensor_grid::interpolation::neighborhood27; break;
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
                case source_mask_choice::compact: sensor_transfer.source_mask = sensor_grid::source_mask_mode::compact; break;
                default: return report_error("sensor source mask must be dense or compact");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::transfer_precision:
                switch (lookup_cli_option(value, transfer_precision_choices)) {
                case transfer_precision_choice::fp32: sensor_transfer.arithmetic = sensor_grid::transfer_arithmetic::fp32; break;
                case transfer_precision_choice::fp64: sensor_transfer.arithmetic = sensor_grid::transfer_arithmetic::fp64; break;
                default: return report_error("sensor transfer precision must be 32 or 64");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::transfer_kernel:
                switch (lookup_cli_option(value, transfer_kernel_choices)) {
                case transfer_kernel_choice::reference: sensor_transfer.specialized = false; break;
                case transfer_kernel_choice::specialized: sensor_transfer.specialized = true; break;
                default: return report_error("sensor transfer must be reference or specialized");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::candidates:
                switch (lookup_cli_option(value, candidate_choices)) {
                case candidate_choice::whole_tiles: sensor_transfer.candidates = sensor_grid::candidate_mode::whole_tiles; break;
                case candidate_choice::voxel_mask: sensor_transfer.candidates = sensor_grid::candidate_mode::voxel_mask; break;
                case candidate_choice::compacted_control: sensor_transfer.candidates = sensor_grid::candidate_mode::compacted_control; break;
                case candidate_choice::compact_fp32: sensor_transfer.candidates = sensor_grid::candidate_mode::compact_fp32; break;
                case candidate_choice::append_fp32: sensor_transfer.candidates = sensor_grid::candidate_mode::append_fp32; break;
                case candidate_choice::compact_fp64: sensor_transfer.candidates = sensor_grid::candidate_mode::compact_fp64; break;
                case candidate_choice::append_fp64: sensor_transfer.candidates = sensor_grid::candidate_mode::append_fp64; break;
                case candidate_choice::coarse_mask: sensor_transfer.candidates = sensor_grid::candidate_mode::coarse_mask; break;
                case candidate_choice::whole_tile_control: sensor_transfer.candidates = sensor_grid::candidate_mode::whole_tile_control; break;
                default: return report_error("invalid sensor candidate method");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::discovery:
                switch (lookup_cli_option(value, discovery_choices)) {
                case discovery_choice::cpu: sensor_transfer.discovery = sensor_grid::discovery_mode::cpu; break;
                case discovery_choice::gpu_host: sensor_transfer.discovery = sensor_grid::discovery_mode::gpu_host; break;
                case discovery_choice::gpu_resident: sensor_transfer.discovery = sensor_grid::discovery_mode::gpu_resident; break;
                default: return report_error("invalid sensor discovery method");
                }
                sensor_options_set = true;
                continue;
            case sensor_mapping_option::candidate_capacity: {
                std::uint32_t capacity{};
                const auto parsed = std::from_chars(value.data(), value.data() + value.size(), capacity);

                if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !capacity || capacity > 536870912)
                    return report_error("invalid sensor candidate capacity");
                sensor_transfer.candidate_capacity = capacity;
                sensor_options_set = true;
                continue;
            }
            default: break;
            }
            const auto resident_parsed = parse_resident_option(option, value, resident_options);

            if (!resident_parsed) return report_error(resident_parsed.error().message);
            if (*resident_parsed) {
                if (lookup_cli_option(option, covoxslam::resident_options) == resident_option::backend) {
                    mapping_enabled = true;
                    cuda_options.enabled = cuda_options.projective = cuda_options.calibrated_policy = false;
                }
                continue;
            }
            const auto policy_parsed = parse_projective_policy_option(option, value, projective_options, projective_policy_option_set);
            if (!policy_parsed) return report_error(policy_parsed.error().message);
            if (*policy_parsed) continue;
            const auto fusion_parsed = parse_ray_fusion_option(option, value, fusion_options);
            if (!fusion_parsed) { std::cerr << fusion_parsed.error().message << '\n'; return 2; }
            if (*fusion_parsed) continue;
            const auto projective_parsed = parse_projective_option(option, value, projective_options, projective_option_set);
            if (!projective_parsed) return report_error(projective_parsed.error().message);
            if (*projective_parsed) {
                if (lookup_cli_option(option, covoxslam::projective_options) != projective_option::candidate_capacity) projective_control_set = true;
                continue;
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
                const auto esdf_parsed = parse_cuda_esdf_option(option, value, esdf_options);

                if (!esdf_parsed) return report_error(esdf_parsed.error().message);
                if (*esdf_parsed) continue;
                const auto cuda_parsed = parse_cuda_mapping_option(option, value, mapping_options, mapping_enabled, cuda_options);

                if (!cuda_parsed) return report_error(cuda_parsed.error().message);
                if (*cuda_parsed) continue;
                const auto parsed = parse_mapping_option(option, value, mapping_options, mapping_enabled);

                if (!parsed) return report_error(parsed.error().message);
                if (!*parsed) return report_error("unsupported submap option");
            }
        }
         if (root.empty()) return report_error("--submaps requires --dataset directory");
        if (nearest_preset && (!sensor_selected || resident_options.selected || cuda_options.enabled ||
            cuda_options.projective || !sensor_grid::matches_combined_nearest({mapping_options, sensor_gather,
                sensor_transfer, sensor_mode, options.ring_stride, options.column_stride})))
            return report_error("explicit options conflict with combined-nearest preset");
        if (sensor_options_set && !sensor_selected) return report_error("sensor options require cuda-sensor-grid");
        if (sensor_selected && sensor_grid::candidate_uses_mask(sensor_transfer.candidates) &&
            (unsigned(sensor_mode) > 2 || sensor_transfer.specialized || sensor_transfer.arithmetic != sensor_grid::transfer_arithmetic::fp64))
            return report_error("candidate masks require reference fp64 point sampling");
        if (sensor_selected && sensor_transfer.arithmetic == sensor_grid::transfer_arithmetic::fp32 && unsigned(sensor_mode) > 2)
            return report_error("fp32 transfer supports nearest, tetrahedral or trilinear sampling");
        if (sensor_selected && sensor_transfer.specialized && (sensor_mode != sensor_grid::interpolation::trilinear ||
            sensor_transfer.arithmetic != sensor_grid::transfer_arithmetic::fp64))
            return report_error("specialized transfer requires fp64 trilinear sampling");
        if (!options.submap_duration_ns || !options.ring_stride || !options.column_stride)
            return report_error("duration and sampling strides must be positive");
        if (!output.empty() && !mapping_enabled)
            return report_error("--output requires --backend cpu-ray, cuda-ray or cuda-projective");
        if (projective_option_set && !cuda_options.projective)
            return report_error("projective options require --backend cuda-projective");
        if (projective_policy_option_set && !cuda_options.calibrated_policy)
            return report_error("projective raycast factor requires --backend cuda-projective-calibrated");
        if (cuda_options.calibrated_policy && projective_control_set)
            return report_error("calibrated policy uses fixed sampling and weighting controls");
        if (cuda_options.projective && cuda_options.contribution_capacity_set)
            return report_error("projective mapping uses --projective-candidate-capacity");
        const auto fusion_selection = validate_ray_fusion_selection(cuda_options.enabled, mapping_enabled, fusion_options);
        if (!fusion_selection) { std::cerr << fusion_selection.error().message << '\n'; return 2; }

        if (resident_options.specified && !resident_options.selected)
            return report_error("resident capacities require a resident backend");
        if (resident_options.warp_specified && (!resident_options.selected || resident_options.mapper.warp.mode == cuda_resident_warp::none))
            return report_error("warp options require a warp backend");
        if (resident_options.free_space_specified && (!resident_options.selected || !resident_options.mapper.free_space.enabled))
            return report_error("free space options require a free space backend");
        if (resident_options.segment_specified && (!resident_options.selected || !resident_options.mapper.segment_length))
            return report_error("segment length requires a segmented backend");
        if (resident_options.selected && (cuda_options.ray_capacity_set || cuda_options.contribution_capacity_set))
            return report_error("use --resident-rays and --resident-contributions with resident mapping");
        pgo_settings.solver.device = cuda_options.mapper.device;
        const bool stored_band = mapping_options.esdf_generator == esdf_backend::cuda_block_band_soa;
        store_settings.device = cuda_options.mapper.device;
        if (stored_band) {
            if (esdf_options.specified) return report_error("block band SoA does not accept blocks execution options");
            store_settings.esdf_generator = esdf_backend::cuda_block_band_soa;
        }
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
        const auto finalize = stored_band ? std::function<void(submap&)>{} :
            cuda_esdf_finalizer(mapping_options, cuda_options.mapper.device, esdf_options);
        auto mapper_options = mapping_options;
        if (finalize || stored_band) mapper_options.esdf_generator = esdf_backend::none;
        std::unique_ptr<cpu_mapper> cpu;
        std::unique_ptr<cuda_mapper> gpu;
        std::unique_ptr<cuda_projective_mapper> projective;
        std::unique_ptr<cuda_resident_context> resident;
        std::optional<operation_error> resident_failure;
        if (mapping_enabled && !sensor_selected) {
            if (resident_options.selected)
            {
                resident_options.mapper.device = cuda_options.mapper.device;
                resident_options.mapper.maximum_device_bytes = cuda_options.mapper.maximum_device_bytes;
                auto created = cuda_resident_context::create(mapper_options, resident_options.mapper);

                if (!created) return report_error(created.error().message);
                resident = std::move(*created);
            }
            else if (cuda_options.projective) {
                projective_options.calibrated_policy = cuda_options.calibrated_policy;
                projective_options.device = cuda_options.mapper.device;
                projective_options.maximum_device_bytes = cuda_options.mapper.maximum_device_bytes;
                if (cuda_options.ray_capacity_set) projective_options.ray_capacity = cuda_options.mapper.ray_capacity;
                projective = std::make_unique<cuda_projective_mapper>(input.sensor, mapper_options, projective_options);
            }
            else if (cuda_options.enabled) gpu = std::make_unique<cuda_mapper>(mapper_options, cuda_options.mapper);
            else cpu = std::make_unique<cpu_mapper>(mapper_options);
        }
        if (gpu) {
            const auto configured = gpu->configure_fusion(fusion_options.mode, fusion_options.short_segment_threshold);
            if (!configured) { std::cerr << configured.error().message << '\n'; return 2; }
        }
        const char* backend = projective ? "cuda-projective" : gpu ? "cuda-ray" : "cpu-ray";
        if (resident) backend = resident_backend_name(resident->fusion_order(), resident->reduction_mode(), resident->uses_step_tasks(), resident->warp_configuration().mode, resident->free_space_configuration().enabled, resident->segment_length() != 0, resident->uses_compact(), resident->uses_compact_batched(), resident->uses_fp32());
        if (cuda_options.calibrated_policy) backend = "cuda-projective-calibrated";
        if (sensor_selected) backend = sensor_gather_name(sensor_gather.backend);
        if (cuda_options.calibrated_policy) std::cout << "projective raycast factor " << projective_options.raycast_subsampling_factor << '\n';
        if (gpu) std::cout << "ray fusion " << cuda_ray_fusion_name(fusion_options.mode)
                           << ", short threshold " << fusion_options.short_segment_threshold
                           << ", maximum rays per batch " << cuda_options.mapper.ray_capacity << '\n';
        std::string backend_label = backend;
        if (nearest_preset) std::cout << "preset combined-nearest, nearest cell only, unknown cells rejected\n";
        if (sensor_selected && sensor_transfer.source_mask == sensor_grid::source_mask_mode::compact)
            backend_label += "-compact-mask";
        if (sensor_selected && sensor_transfer.arithmetic == sensor_grid::transfer_arithmetic::fp32)
            backend_label += "-fp32-transfer";
        if (sensor_selected && sensor_transfer.specialized) backend_label += "-specialized-transfer";
        if (sensor_selected && sensor_transfer.candidates == sensor_grid::candidate_mode::voxel_mask) backend_label += "-voxel-mask";
        if (sensor_selected && sensor_transfer.candidates == sensor_grid::candidate_mode::compacted_control) backend_label += "-compacted-control";
        if (sensor_selected && sensor_transfer.candidates == sensor_grid::candidate_mode::compact_fp32) backend_label += "-compact-fp32";
        if (sensor_selected && sensor_transfer.candidates == sensor_grid::candidate_mode::append_fp32) backend_label += "-append-fp32";
        if (sensor_selected && sensor_transfer.candidates == sensor_grid::candidate_mode::compact_fp64) backend_label += "-compact-fp64";
        if (sensor_selected && sensor_transfer.candidates == sensor_grid::candidate_mode::append_fp64) backend_label += "-append-fp64";
        if (sensor_selected && sensor_transfer.candidates == sensor_grid::candidate_mode::coarse_mask) backend_label += "-coarse-mask";
        if (sensor_selected && sensor_transfer.candidates == sensor_grid::candidate_mode::whole_tile_control) backend_label += "-whole-tile-control";
        if (sensor_selected && sensor_transfer.discovery == sensor_grid::discovery_mode::gpu_host) backend_label += "-gpu-discovery";
        if (sensor_selected && sensor_transfer.discovery == sensor_grid::discovery_mode::gpu_resident) backend_label += "-gpu-resident";
        if (store_settings.resident_mapping) backend_label += "-soa";
        const auto fusion_label = sensor_selected ? sensor_sampling_name(sensor_mode) : resident ? resident_fusion_name(resident->fusion_order(), resident->reduction_mode(), resident->uses_step_tasks(), resident->warp_configuration().mode, resident->free_space_configuration().enabled, resident->segment_length() != 0, resident->uses_compact(), resident->uses_compact_batched(), resident->uses_fp32()) : "unspecified";
        auto exporter = pgo_settings.solver.enabled || store_settings.enabled ? std::unique_ptr<map_exporter>{} :
            open_map_export(output, input, mapping_options, backend_label.c_str(), fusion_label);
        std::vector<submap> pgo_maps;
        std::unique_ptr<resident_pipeline> resident_maps;
        if (store_settings.resident_mapping) {
            auto created = resident_pipeline::create(store_settings, pgo_settings.solver);
            if (!created) return report_error(created.error().message);
            resident_maps = std::move(*created);
            std::cout << "mapping storage resident-soa, device handoff, deferred canonical readback\n";
        }
        const auto publish = [&](const submap& completed) {
            if (pgo_settings.solver.enabled || store_settings.enabled) pgo_maps.push_back(completed);
            else if (exporter) exporter->write(completed);
        };
        auto callbacks = resident ? resident_callbacks(*resident, mapping_options, resident_failure, publish, finalize) :
            projective ? mapping_callbacks(*projective, publish, finalize) :
            gpu ? mapping_callbacks(*gpu, publish, finalize) :
            cpu ? mapping_callbacks(*cpu, publish, finalize) : pipeline_callbacks{};
        if (sensor_selected) {
            auto selected = sensor_mapping_callbacks(input.sensor, mapper_options, cuda_options.mapper.device,
                resident_failure, publish, finalize, sensor_mode, sensor_gather, cuda_options.mapper.maximum_device_bytes,
                sensor_transfer, resident_maps.get());
            if (!selected) return report_error(selected.error().message);
            callbacks = std::move(*selected);
            std::cout << "sensor field uses calibrated ranges and scan start poses\n";
        }
        const auto result = run_frame_pipeline(input, options, callbacks);

        if (result.cancelled) return report_error(resident_failure ? resident_failure->message : "mapping cancelled");
        if (pgo_settings.solver.enabled || store_settings.enabled) {
            auto optimized = [&]() -> operation_result<pgo_cli_result> {
                if (!store_settings.enabled) return finish_pgo(pgo_maps, pgo_settings);
                auto stored = resident_maps ? resident_maps->finish(pgo_maps, pgo_settings.solver, pgo_settings.shadow) :
                    finish_stored_pipeline(pgo_maps, store_settings, pgo_settings.solver, pgo_settings.shadow);
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
