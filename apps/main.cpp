#include <covoxslam/input/dataset.hpp>
#include <covoxslam/core/result.hpp>
#include "frame_pipeline_cli.hpp"
#include "sensor_cli.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <iostream>
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
struct inspection_cli_defaults
{
    covoxslam::dataset_load_options input;
    std::size_t ring_stride{covoxslam::pipeline_options{}.ring_stride};
    std::size_t column_stride{covoxslam::pipeline_options{}.column_stride};
    covoxslam::point_source source{covoxslam::point_source::raw};
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
    constexpr std::string_view fallback{"invalid"};
    const covoxslam::sensor_grid::mapping_preset preset;
    const covoxslam::pipeline_options pipeline_defaults;
    const covoxslam::store_pipeline_options store_defaults;
    const covoxslam::pgo_cli_options pgo_defaults;
    const inspection_cli_defaults inspection_defaults;
    const auto mapping_storage = store_defaults.resident_mapping ? covoxslam::store_value::resident_soa :
        covoxslam::store_value::unknown;
    const auto store_layout = store_defaults.layout == covoxslam::storage::layout::pool_soa ?
        covoxslam::store_value::soa : covoxslam::store_value::unknown;
    const auto esdf = preset.mapping.esdf_generator == covoxslam::esdf_backend::cuda_block_band_soa ?
        covoxslam::esdf_value::block_band_soa : covoxslam::esdf_value::unknown;
    const auto backend = covoxslam::sensor_grid::combined_gather(preset.gather.backend) ?
        covoxslam::mapping_backend_choice::combined : covoxslam::mapping_backend_choice::unknown;
    const auto sampling = preset.sampling == covoxslam::sensor_grid::interpolation::nearest ?
        covoxslam::sensor_sampling_value::nearest : covoxslam::sensor_sampling_value::unknown;
    const auto gather = covoxslam::sensor_grid::combined_gather(preset.gather.backend) ?
        covoxslam::gather_choice::combined : covoxslam::gather_choice::unknown;
    const auto execution = preset.gather.execution == covoxslam::sensor_grid::execution_mode::stream ?
        covoxslam::execution_choice::stream : covoxslam::execution_choice::unknown;
    const auto source_mask = preset.transfer.source_mask == covoxslam::sensor_grid::source_mask_mode::dense ?
        covoxslam::source_mask_choice::dense : covoxslam::source_mask_choice::unknown;
    const auto candidates = preset.transfer.candidates == covoxslam::sensor_grid::candidate_mode::whole_tiles ?
        covoxslam::candidate_choice::whole_tiles : covoxslam::candidate_choice::unknown;
    const auto discovery = preset.transfer.discovery == covoxslam::sensor_grid::discovery_mode::gpu_resident ?
        covoxslam::discovery_choice::gpu_resident : covoxslam::discovery_choice::unknown;
    const auto transfer = preset.transfer.specialized ? covoxslam::transfer_kernel_choice::unknown :
        covoxslam::transfer_kernel_choice::reference;
    const auto precision = preset.transfer.arithmetic == covoxslam::sensor_grid::transfer_arithmetic::fp64 ?
        covoxslam::transfer_precision_choice::fp64 : covoxslam::transfer_precision_choice::unknown;
    const auto trajectory = pipeline_defaults.pose_source == covoxslam::pose_source::odometry ?
        covoxslam::trajectory_option::odometry : covoxslam::trajectory_option::ground_truth;
    const auto verification = inspection_defaults.input.verification ==
        covoxslam::dataset_verification_mode::prefer_cache ? covoxslam::verification_option::cached :
        covoxslam::verification_option::full;
    const auto negative_band = store_defaults.negative_band ? covoxslam::store_boolean_value::true_value :
        covoxslam::store_boolean_value::false_value;
    const auto pgo = pgo_defaults.solver.enabled ? covoxslam::pgo_value::unknown : covoxslam::pgo_value::none;
    auto fields = covoxslam::store_value::unknown;
    switch (store_defaults.fields) {
    case covoxslam::stored_pgo_fields::copied: fields = covoxslam::store_value::copied; break;
    case covoxslam::stored_pgo_fields::borrowed: fields = covoxslam::store_value::borrowed; break;
    case covoxslam::stored_pgo_fields::resident: fields = covoxslam::store_value::resident; break;
    }
    const auto point = inspection_defaults.source == covoxslam::point_source::raw ? point_option::raw :
        point_option::deskewed;
    std::cout << "coVoxSLAM\n"
        << "Usage: covoxslam --submaps --dataset PATH [--output PATH]\n"
        << "  --preset " << covoxslam::combined_nearest_preset_name
        << "              Default and sole pipeline\n"
        << "  --mapping-storage " << covoxslam::cli_option_name(mapping_storage,
            covoxslam::store_values, fallback) << " --map-store "
        << covoxslam::cli_option_name(store_layout, covoxslam::store_values, fallback) << '\n'
        << "  --esdf-generator " << covoxslam::cli_option_name(esdf, covoxslam::esdf_values, fallback) << '\n'
        << "  --backend " << covoxslam::cli_option_name(backend, covoxslam::mapping_backend_choices, fallback)
        << " --sensor-grid-interpolation "
        << covoxslam::cli_option_name(sampling, covoxslam::sensor_sampling_values, fallback) << '\n'
        << "  --sensor-grid-gather " << covoxslam::cli_option_name(gather, covoxslam::gather_choices, fallback)
        << " --sensor-grid-execution "
        << covoxslam::cli_option_name(execution, covoxslam::execution_choices, fallback) << '\n'
        << "  --sensor-grid-short-list-max " << preset.gather.short_list_max
        << " --sensor-grid-block-list-min " << preset.gather.block_list_min << '\n'
        << "  --sensor-grid-source-mask "
        << covoxslam::cli_option_name(source_mask, covoxslam::source_mask_choices, fallback)
        << " --sensor-grid-candidates "
        << covoxslam::cli_option_name(candidates, covoxslam::candidate_choices, fallback) << '\n'
        << "  --sensor-grid-discovery "
        << covoxslam::cli_option_name(discovery, covoxslam::discovery_choices, fallback)
        << " --sensor-grid-transfer "
        << covoxslam::cli_option_name(transfer, covoxslam::transfer_kernel_choices, fallback) << '\n'
        << "  --sensor-grid-transfer-precision "
        << covoxslam::cli_option_name(precision, covoxslam::transfer_precision_choices, fallback) << '\n'
        << "  --ring-stride " << preset.ring_stride << " --column-stride " << preset.column_stride
        << " --voxel-size " << preset.mapping.voxel_size << '\n'
        << "  --truncation-distance " << preset.mapping.truncation_distance
        << " --maximum-ray-length " << preset.mapping.maximum_ray_length_m << '\n'
        << "  --cuda-device N --cuda-device-bytes N --maximum-tiles N\n"
        << "  --maximum-frames N --maximum-submaps N   Zero means unlimited\n"
        << "  --submap-duration-ns N                  Default " << pipeline_defaults.submap_duration_ns << '\n'
        << "  --pose-source " << covoxslam::cli_option_name(trajectory,
            covoxslam::trajectory_options, fallback) << '|'
        << covoxslam::cli_option_name(covoxslam::trajectory_option::ground_truth,
            covoxslam::trajectory_options, fallback) << " --verification "
        << covoxslam::cli_option_name(verification, covoxslam::verification_options, fallback) << '|'
        << covoxslam::cli_option_name(covoxslam::verification_option::full,
            covoxslam::verification_options, fallback) << '\n'
        << "  --esdf-site-band M --esdf-maximum-distance M --esdf-minimum-weight W\n"
        << "  --store-negative-band " << covoxslam::cli_option_name(negative_band,
            covoxslam::store_boolean_values, fallback) << '|'
        << covoxslam::cli_option_name(covoxslam::store_boolean_value::true_value,
            covoxslam::store_boolean_values, fallback) << '\n'
        << "  --store-output PATH --store-memory-mib N\n"
        << "  --pgo " << covoxslam::cli_option_name(pgo, covoxslam::pgo_values, fallback) << '|'
        << covoxslam::cli_option_name(covoxslam::pgo_value::cpu, covoxslam::pgo_values, fallback) << '|'
        << covoxslam::cli_option_name(covoxslam::pgo_value::cuda, covoxslam::pgo_values, fallback)
        << " --pgo-fields " << covoxslam::cli_option_name(fields, covoxslam::store_values, fallback) << '|'
        << covoxslam::cli_option_name(covoxslam::store_value::borrowed, covoxslam::store_values, fallback) << '|'
        << covoxslam::cli_option_name(covoxslam::store_value::resident, covoxslam::store_values, fallback) << '\n'
        << "    Borrowed fields require CPU PGO, resident fields require CUDA PGO\n"
        << "  --dataset PATH [--frame INDEX]           Inspect input metadata or points\n"
        << "  --point-source " << covoxslam::cli_option_name(point, point_options, fallback) << '|'
        << covoxslam::cli_option_name(point_option::deskewed, point_options, fallback)
        << "              Default " << covoxslam::cli_option_name(point, point_options, fallback) << '\n'
        << "  --sensor-grid --dataset PATH            Inspect calibrated lookup\n"
        << "  Retired option names are recognized and rejected explicitly\n"
        << "  --help | --version\n";
}

int run_inspection_cli(int argc, char** argv);
}

int main(int argc, char** argv)
{
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
        return run_inspection_cli(argc, argv);
}

namespace
{
int run_inspection_cli(int argc, char** argv)
{
        const auto report_error = [](std::string_view message) {
            std::cerr << "covoxslam: " << message << '\n';
            return 2;
        };
        std::filesystem::path dataset_path;
        const inspection_cli_defaults defaults;
        auto input_options = defaults.input;
        std::size_t frame_index{}, ring_stride{defaults.ring_stride}, column_stride{defaults.column_stride};
        bool frame_selected{}, sampling_selected{};
        auto source = defaults.source;
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
        if (!frame) return report_error(frame.error().message);
        const auto& points = options.raw_xyz ? frame->raw_xyz : frame->deskewed_xyz;
        if (points.empty()) return report_error("requested point source is unavailable");
        const auto rings = sampled_extent(frame->descriptor.rows, ring_stride);
         const auto columns = sampled_extent(frame->descriptor.columns, column_stride);
        std::size_t selected{}, valid{};
        for (std::size_t column_sample = 0; column_sample < columns; ++column_sample)
        {
            const auto column = column_sample * column_stride;
            for (std::size_t ring_sample = 0; ring_sample < rings; ++ring_sample) {
                const auto ring = ring_sample * ring_stride;
                ++selected;
                const auto returned = frame->valid_return(ring, column);
                if (!returned) return report_error(returned.error().message);
                if (!*returned) continue;
                const auto point = frame->point(source, ring, column);
                if (!point) return report_error(point.error().message);
                 if (std::isfinite(point->get().x) && std::isfinite(point->get().y) &&
                     std::isfinite(point->get().z)) ++valid;
            }
        }
        std::cout << "frame " << frame_index << " id " << frame->descriptor.frame_id << '\n'
                  << "shape " << frame->descriptor.rows << ' ' << frame->descriptor.columns << '\n'
                  << "selected beams " << selected << '\n' << "valid selected returns " << valid << '\n';
        if (columns != 0)
        {
            const auto first = frame->beam_stamp_ns(0);
            if (!first) return report_error(first.error().message);
            const auto last = frame->beam_stamp_ns((columns - 1) * column_stride);
            if (!last) return report_error(last.error().message);
            std::cout << "selected column stamps " << *first << ' ' << *last << '\n';
        }
        return 0;
}
}
