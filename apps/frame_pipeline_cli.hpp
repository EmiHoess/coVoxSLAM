#pragma once

#include <covoxslam/mapping.hpp>
#include "projective_policy_cli.hpp"
#include "ray_fusion_cli.hpp"
#include "cuda_esdf_cli.hpp"
#include "map_export_cli.hpp"
#include "resident_cli.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace covoxslam
{

enum class submap_cli_option { unknown, dataset, output, frames, submaps, rings, columns, duration, verification, pose_source };
inline constexpr cli_option_entry<submap_cli_option> submap_cli_options[]{
    {"--dataset", submap_cli_option::dataset}, {"--output", submap_cli_option::output},
    {"--maximum-frames", submap_cli_option::frames}, {"--maximum-submaps", submap_cli_option::submaps},
     {"--ring-stride", submap_cli_option::rings}, {"--column-stride", submap_cli_option::columns},
    {"--submap-duration-ns", submap_cli_option::duration}, {"--verification", submap_cli_option::verification},
    {"--pose-source", submap_cli_option::pose_source}
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
        for (int index = 2; index < argc; index += 2)
        {
            const std::string_view option(argv[index]);
            if (index + 1 >= argc) return report_error("missing submap option value");
            const std::string_view value(argv[index + 1]);
            const auto resident_parsed = parse_resident_option(option, value, resident_options);

            if (!resident_parsed) return report_error(resident_parsed.error().message);
            if (*resident_parsed) {
                if (option == "--backend") {
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
                if (option != "--projective-candidate-capacity") projective_control_set = true;
                continue;
            }
            switch (lookup_cli_option(option, submap_cli_options))
            {
            case submap_cli_option::dataset: root = argv[index + 1]; continue;
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
            if (option == "--verification") {
                if (value == "cached") input_options.verification = dataset_verification_mode::prefer_cache;
                else if (value == "full") input_options.verification = dataset_verification_mode::force_full;
                else return report_error("unsupported input verification mode");
            }
            else if (option == "--pose-source")
            {
                if (value == "odometry") options.pose_source = pose_source::odometry;
                else if (value == "ground_truth") options.pose_source = pose_source::ground_truth;
                else return report_error("unsupported trajectory source");
            }
            else
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
        if (resident_options.selected && (cuda_options.ray_capacity_set || cuda_options.contribution_capacity_set))
            return report_error("use --resident-rays and --resident-contributions with resident mapping");
        const auto input = load_dataset(root, input_options);
        const auto finalize = cuda_esdf_finalizer(mapping_options, cuda_options.mapper.device, esdf_options);
        auto mapper_options = mapping_options;
        if (finalize) mapper_options.esdf_generator = esdf_backend::none;
        std::unique_ptr<cpu_mapper> cpu;
        std::unique_ptr<cuda_mapper> gpu;
        std::unique_ptr<cuda_projective_mapper> projective;
        std::unique_ptr<cuda_resident_context> resident;
        std::optional<operation_error> resident_failure;
        if (mapping_enabled) {
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
        if (resident) backend = resident_backend_name(resident->fusion_order(), resident->reduction_mode(), resident->uses_step_tasks(), resident->warp_configuration().mode, resident->free_space_configuration().enabled);
        if (cuda_options.calibrated_policy) backend = "cuda-projective-calibrated";
        if (cuda_options.calibrated_policy) std::cout << "projective raycast factor " << projective_options.raycast_subsampling_factor << '\n';
        if (gpu) std::cout << "ray fusion " << cuda_ray_fusion_name(fusion_options.mode)
                           << ", short threshold " << fusion_options.short_segment_threshold
                           << ", maximum rays per batch " << cuda_options.mapper.ray_capacity << '\n';
        auto exporter = open_map_export(output, input, mapping_options, backend,
            resident ? resident_fusion_name(resident->fusion_order(), resident->reduction_mode(), resident->uses_step_tasks(), resident->warp_configuration().mode, resident->free_space_configuration().enabled) : "unspecified");
        const auto publish = [&](const submap& completed) { if (exporter) exporter->write(completed); };
        const auto callbacks = resident ? resident_callbacks(*resident, mapping_options, resident_failure, publish, finalize) :
            projective ? mapping_callbacks(*projective, publish, finalize) :
            gpu ? mapping_callbacks(*gpu, publish, finalize) :
            cpu ? mapping_callbacks(*cpu, publish, finalize) : pipeline_callbacks{};
        const auto result = run_frame_pipeline(input, options, callbacks);

        if (result.cancelled) return report_error(resident_failure ? resident_failure->message : "mapping cancelled");
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
        return 0;
}

}
