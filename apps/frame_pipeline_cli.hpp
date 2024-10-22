#pragma once

#include <covoxslam/mapping.hpp>
#include "mapping_cli.hpp"
#include "map_export_cli.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace covoxslam
{

inline int run_submap_cli(int argc, char** argv)
{
    try {
        pipeline_options options;
        mapping_config mapping_options;
        bool mapping_enabled{};
        dataset_load_options input_options;
        std::filesystem::path root;
        std::filesystem::path output;
         const auto integer = [](std::string_view text) {
            std::uint64_t result{};
            const auto converted = std::from_chars(text.data(), text.data() + text.size(), result);
            if (converted.ec != std::errc{} || converted.ptr != text.data() + text.size())
                throw std::invalid_argument("expected an unsigned decimal integer");
            return result;
        }
        ;
        const auto count = [&](std::string_view text) {
            const auto result = integer(text);
            if (result > std::numeric_limits<std::size_t>::max())
                throw std::invalid_argument("option exceeds addressable size");
             return static_cast<std::size_t>(result);
        };
        for (int index = 2; index < argc; index += 2)
        {
            const std::string_view option(argv[index]);
            if (index + 1 >= argc) throw std::invalid_argument("missing submap option value");
            const std::string_view value(argv[index + 1]);
            if (option == "--dataset") root = argv[index + 1];
            else if (option == "--output") {
                if (value.empty()) throw std::invalid_argument("--output requires a directory");
                 output = argv[index + 1];
            }
            else if (option == "--maximum-frames") options.maximum_frames = count(value);
            else if (option == "--maximum-submaps") options.maximum_submaps = count(value);
             else if (option == "--ring-stride") options.ring_stride = count(value);
            else if (option == "--column-stride") options.column_stride = count(value);
            else if (option == "--submap-duration-ns") options.submap_duration_ns = integer(value);
            else if (option == "--verification") {
                if (value == "cached") input_options.verification = dataset_verification_mode::prefer_cache;
                else if (value == "full") input_options.verification = dataset_verification_mode::force_full;
                else throw std::invalid_argument("unsupported input verification mode");
            }
            else if (option == "--pose-source")
            {
                if (value == "odometry") options.pose_source = pose_source::odometry;
                else if (value == "ground_truth") options.pose_source = pose_source::ground_truth;
                else throw std::invalid_argument("unsupported trajectory source");
            }
            else if (!parse_mapping_option(option, value, mapping_options, mapping_enabled))
                throw std::invalid_argument("unsupported submap option");
        }
         if (root.empty()) throw std::invalid_argument("--submaps requires --dataset directory");
        if (!options.submap_duration_ns || !options.ring_stride || !options.column_stride)
            throw std::invalid_argument("duration and sampling strides must be positive");
        if (!output.empty() && !mapping_enabled)
            throw std::invalid_argument("--output requires --backend cpu-ray");
        const auto input = load_dataset(root, input_options);
        cpu_mapper mapper(mapping_options);
        auto exporter = open_map_export(output, input, mapping_options);
        const auto callbacks = mapping_enabled ? cpu_mapping_callbacks(mapper, [&](const submap& completed) {
            if (exporter) exporter->write(completed);
        }) : pipeline_callbacks{};
        const auto result = run_frame_pipeline(input, options, callbacks);
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
    catch (const std::invalid_argument& error) {
        std::cerr << "covoxslam: " << error.what() << '\n';
        return 2;
    }
    catch (const std::exception& error)
    {
         std::cerr << "covoxslam: " << error.what() << '\n';
        return 1;
    }
}

}
