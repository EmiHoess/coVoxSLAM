#include <covoxslam/dataset.hpp>
#include "frame_pipeline_cli.hpp"

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
std::size_t parse_index(std::string_view value, std::string_view option)
{
    std::size_t result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
     if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument(std::string(option) + " needs a nonnegative integer");
    return result;
}

std::size_t sampled_extent(std::size_t extent, std::size_t stride) {
    return extent / stride + (extent % stride != 0);
}

void usage()
{
    std::cout << "Usage: covoxslam --dataset PATH [--frame INDEX]\n"
              << "  --ring-stride N          Positive ring stride, default 1\n"
              << "  --column-stride N        Positive column stride, default 1\n"
              << "  --point-source raw|deskewed  Default raw\n"
              << "  --verification cached|full  Default cached\n"
              << "  --submaps --dataset PATH --backend cpu-ray  Build local TSDF maps\n"
              << "  --voxel-size M --truncation-distance M --maximum-ray-length M\n"
              << "  --esdf-generator none|cpu-voxblox-batch  Default cpu-voxblox-batch\n"
              << "  --submaps --dataset PATH  Deskew scans and report submap metadata\n"
              << "    --maximum-frames N | --maximum-submaps N  Zero means unlimited\n"
              << "    --submap-duration-ns N  Default 10000000000\n"
              << "    --pose-source odometry|ground_truth  Default odometry\n"
               << "  --frame INDEX            Zero based frame index\n"
              << "  --help | --version\n";
}
}

int main(int argc, char** argv)
{
    try
    {
        if (argc >= 2 && std::string_view(argv[1]) == "--submaps")
            return covoxslam::run_submap_cli(argc, argv);
        if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
            usage();
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--version")
        {
            std::cout << "covoxslam " << covoxslam_version << '\n';
             return 0;
        }
        std::filesystem::path dataset_path;
        covoxslam::dataset_load_options input_options;
        std::size_t frame_index{}, ring_stride{1}, column_stride{1};
        bool frame_selected{}, sampling_selected{};
        auto source = covoxslam::point_source::raw;
        for (int index = 1; index < argc; ++index)
        {
            const std::string_view option = argv[index];
            if (option != "--dataset" && option != "--frame" && option != "--ring-stride" &&
                option != "--column-stride" && option != "--point-source" && option != "--verification")
                throw std::invalid_argument("unknown option: " + std::string(option));
            if (++index >= argc)
                throw std::invalid_argument(std::string(option) + " needs a value");
             const std::string_view value = argv[index];
            if (option == "--dataset") dataset_path = std::string(value);
            else if (option == "--verification")
            {
                if (value != "cached" && value != "full")
                    throw std::invalid_argument("--verification needs cached or full");
                input_options.verification = value == "cached" ? covoxslam::dataset_verification_mode::prefer_cache
                    : covoxslam::dataset_verification_mode::force_full;
            }
            else if (option == "--frame") {
                frame_index = parse_index(value, option);
                frame_selected = true;
            }
            else if (option == "--point-source")
            {
                if (value != "raw" && value != "deskewed")
                    throw std::invalid_argument("--point-source needs raw or deskewed");
                source = value == "raw" ? covoxslam::point_source::raw : covoxslam::point_source::deskewed;
                sampling_selected = true;
            }
            else
              {
                const auto stride = parse_index(value, option);
                if (stride == 0) throw std::invalid_argument(std::string(option) + " must be positive");
                if (option == "--ring-stride") ring_stride = stride;
                else column_stride = stride;
                sampling_selected = true;
            }
        }
        if (dataset_path.empty()) throw std::invalid_argument("--dataset is required");
        if (sampling_selected && !frame_selected)
            throw std::invalid_argument("frame sampling options require --frame");
         const auto dataset = covoxslam::load_dataset(dataset_path, input_options);
        std::cout << "dataset " << dataset.name << '\n'
                  << "frames " << dataset.frames.size() << '\n'
                  << "sensor shape " << dataset.sensor.rings << ' ' << dataset.sensor.columns << '\n';
        std::cout << "input source " << (dataset.input_source == covoxslam::dataset_input_source::validated_cache
            ? "validated_cache" : "full_scan") << '\n' << "input hash " << dataset.content_fingerprint << '\n';
        if (!frame_selected) return 0;
        if (frame_index >= dataset.frames.size()) throw std::out_of_range("frame index is out of range");
        covoxslam::frame_load_options options{};
        options.raw_xyz = source == covoxslam::point_source::raw;
        options.deskewed_xyz = source == covoxslam::point_source::deskewed;
        const auto frame = covoxslam::load_frame(dataset, frame_index, options);
        const auto& points = options.raw_xyz ? frame.raw_xyz : frame.deskewed_xyz;
        if (points.empty()) throw std::runtime_error("requested point source is unavailable");
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
