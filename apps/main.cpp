#include <covoxslam/dataset.hpp>
#include <covoxslam/result.hpp>
#include "frame_pipeline_cli.hpp"
#include "sensor_grid_cli.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace
{
enum class application_option { unknown, submaps, sensor_grid, help, version };
constexpr covoxslam::cli_option_entry<application_option> application_options[]{
    {"--submaps", application_option::submaps}, {"--help", application_option::help},
     {"--version", application_option::version}
    , {"--sensor-grid", application_option::sensor_grid}
};
enum class inspection_option { unknown, dataset, frame, ring_stride, column_stride, point_source, verification };
constexpr covoxslam::cli_option_entry<inspection_option> inspection_options[]{
    {"--dataset", inspection_option::dataset}, {"--frame", inspection_option::frame},
    {"--ring-stride", inspection_option::ring_stride}, {"--column-stride", inspection_option::column_stride},
     {"--point-source", inspection_option::point_source}, {"--verification", inspection_option::verification}
};
enum class point_option { unknown, raw, deskewed };
constexpr covoxslam::cli_option_entry<point_option> point_options[]{
    {"raw", point_option::raw}, {"deskewed", point_option::deskewed}
};

covoxslam::operation_result<std::size_t> parse_index(std::string_view value, std::string_view option)
{
    std::size_t result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
     if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        return std::unexpected(covoxslam::operation_error{std::string(option) + " needs a nonnegative integer"});
    return result;
}

std::size_t sampled_extent(std::size_t extent, std::size_t stride) {
    return extent / stride + (extent % stride != 0);
}

void usage()
{
    std::cout << "coVoxSLAM\n"
        << "Usage: covoxslam --submaps --dataset PATH [--output PATH]\n"
        << "  --preset combined-nearest              Default and sole pipeline\n"
        << "  --mapping-storage resident-soa --map-store soa\n"
        << "  --esdf-generator cuda-block-band-soa\n"
        << "  --backend cuda-sensor-grid-combined --sensor-grid-interpolation nearest\n"
        << "  --sensor-grid-gather hybrid-combined --sensor-grid-execution stream\n"
        << "  --sensor-grid-short-list-max 16 --sensor-grid-block-list-min 1024\n"
        << "  --sensor-grid-source-mask dense --sensor-grid-candidates whole-tiles\n"
        << "  --sensor-grid-discovery gpu-resident --sensor-grid-transfer reference\n"
        << "  --sensor-grid-transfer-precision 64\n"
        << "  --ring-stride 1 --column-stride 1 --voxel-size 0.2\n"
        << "  --truncation-distance 0.6 --maximum-ray-length 16\n"
        << "  --cuda-device N --cuda-device-bytes N --maximum-tiles N\n"
        << "  --maximum-frames N --maximum-submaps N   Zero means unlimited\n"
        << "  --submap-duration-ns N                  Default 10000000000\n"
        << "  --pose-source odometry|ground_truth --verification cached|full\n"
        << "  --esdf-site-band M --esdf-maximum-distance M --esdf-minimum-weight W\n"
        << "  --store-negative-band false|true\n"
        << "  --store-output PATH --store-memory-mib N\n"
        << "  --pgo none|cpu|cuda --pgo-fields copied|borrowed|resident\n"
        << "    Borrowed fields require CPU PGO, resident fields require CUDA PGO\n"
        << "  --dataset PATH [--frame INDEX]           Inspect input metadata or points\n"
        << "  --point-source raw|deskewed              Default raw\n"
        << "  --sensor-grid --dataset PATH            Inspect calibrated lookup\n"
        << "  Retired option names are recognized and rejected explicitly\n"
        << "  --help | --version\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        const auto report_error = [](std::string_view message) {
            std::cerr << "covoxslam: " << message << '\n';
            return 2;
        };
        const auto mode = argc == 1 ? application_option::help : covoxslam::lookup_cli_option(argv[1], application_options);
        switch (mode)
        {
        case application_option::submaps: return covoxslam::run_submap_cli(argc, argv);
        case application_option::sensor_grid: return covoxslam::run_sensor_grid_cli(argc, argv);
        case application_option::help:
            if (argc <= 2) { usage(); return 0; }
            break;
        case application_option::version:

            if (argc == 2) {
                std::cout << "covoxslam " << covoxslam_version << '\n';
                 return 0;
            }
            break;
        default: break;
        }
        std::filesystem::path dataset_path;
        covoxslam::dataset_load_options input_options;
        std::size_t frame_index{}, ring_stride{1}, column_stride{1};
        bool frame_selected{}, sampling_selected{};
        auto source = covoxslam::point_source::raw;
        for (int index = 1; index < argc; ++index)
        {
            const std::string_view option = argv[index];
            if (covoxslam::lookup_cli_option(option, inspection_options) == inspection_option::unknown)
                return report_error("unknown option: " + std::string(option));
            if (++index >= argc)
                return report_error(std::string(option) + " needs a value");
             const std::string_view value = argv[index];
            switch (covoxslam::lookup_cli_option(option, inspection_options)) {
            case inspection_option::dataset: dataset_path = std::string(value); continue;
            case inspection_option::frame:
            {
                const auto parsed = parse_index(value, option);

                if (!parsed) return report_error(parsed.error().message);
                frame_index = *parsed;
                 frame_selected = true;
                continue;
            }
            case inspection_option::verification:
            {
                switch (covoxslam::lookup_cli_option(value, covoxslam::verification_options)) {
                case covoxslam::verification_option::cached: input_options.verification = covoxslam::dataset_verification_mode::prefer_cache; break;
                case covoxslam::verification_option::full: input_options.verification = covoxslam::dataset_verification_mode::force_full; break;
                default: return report_error("--verification needs cached or full");
                }
                continue;
            }
            case inspection_option::point_source:
            {
                switch (covoxslam::lookup_cli_option(value, point_options)) {
                case point_option::raw: source = covoxslam::point_source::raw; break;
                case point_option::deskewed: source = covoxslam::point_source::deskewed; break;
                default: return report_error("--point-source needs raw or deskewed");
                }
                sampling_selected = true;
                break;
            }
            case inspection_option::ring_stride:
            case inspection_option::column_stride:
              {
                const auto stride = parse_index(value, option);
                if (!stride || *stride == 0) {
                    const auto diagnostic = !stride ? stride.error().message : std::string(option) + " must be positive";
                    std::cerr << "covoxslam: " << diagnostic << '\n';
                    return 2;
                }
                switch (covoxslam::lookup_cli_option(option, inspection_options)) {
                case inspection_option::ring_stride: ring_stride = *stride; break;
                default: column_stride = *stride; break;
                }
                sampling_selected = true;
                break;
            }
            default: return report_error("unknown option: " + std::string(option));
            }
        }
        if (dataset_path.empty()) return report_error("--dataset is required");
        if (sampling_selected && !frame_selected)
            return report_error("frame sampling options require --frame");
         const auto loaded = covoxslam::load_dataset(dataset_path, input_options);
        if (!loaded) return report_error(loaded.error().message);
        const auto& dataset = *loaded;
        std::cout << "dataset " << dataset.name << '\n'
                  << "frames " << dataset.frames.size() << '\n'
                  << "sensor shape " << dataset.sensor.rings << ' ' << dataset.sensor.columns << '\n';
        std::cout << "input source " << (dataset.input_source == covoxslam::dataset_input_source::validated_cache
            ? "validated_cache" : "full_scan") << '\n' << "input hash " << dataset.content_fingerprint << '\n';
        if (!frame_selected) return 0;
        if (frame_index >= dataset.frames.size()) return report_error("frame index is out of range");
        covoxslam::frame_load_options options{};
        options.raw_xyz = source == covoxslam::point_source::raw;
        options.deskewed_xyz = source == covoxslam::point_source::deskewed;
        const auto frame = covoxslam::load_frame(dataset, frame_index, options);
        const auto& points = options.raw_xyz ? frame.raw_xyz : frame.deskewed_xyz;
        if (points.empty()) return report_error("requested point source is unavailable");
        const auto rings = sampled_extent(frame.descriptor.rows, ring_stride);
         const auto columns = sampled_extent(frame.descriptor.columns, column_stride);
        std::size_t selected{}, valid{};
        for (std::size_t column_sample = 0; column_sample < columns; ++column_sample)
        {
            const auto column = column_sample * column_stride;
            for (std::size_t ring_sample = 0; ring_sample < rings; ++ring_sample) {
                const auto ring = ring_sample * ring_stride;
                ++selected;
                if (!frame.valid_return(ring, column)) continue;
                const auto point = frame.point(source, ring, column);
                 if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z)) ++valid;
            }
        }
        std::cout << "frame " << frame_index << " id " << frame.descriptor.frame_id << '\n'
                  << "shape " << frame.descriptor.rows << ' ' << frame.descriptor.columns << '\n'
                  << "selected beams " << selected << '\n' << "valid selected returns " << valid << '\n';
        if (columns != 0)
        {
            std::cout << "selected column stamps " << frame.beam_stamp_ns(0) << ' '
                      << frame.beam_stamp_ns((columns - 1) * column_stride) << '\n';
        }
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "covoxslam: " << error.what() << '\n';
         return 2;
    }
}
