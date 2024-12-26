#pragma once
#include <covoxslam/sensor_grid_precision.h>

#include <covoxslam/sensor_grid_calibration.hpp>
#include <covoxslam/sensor_grid_cuda.hpp>
#include "cli_lookup.hpp"
#include "sensor_gather_cli.hpp"
#include <charconv>
#include <chrono>
#include <iostream>

namespace covoxslam
{
enum class sensor_option { unknown, dataset, radius, size, factor, truncation, frames, cap, first, backend, device, gather, threshold };
inline constexpr cli_option_entry<sensor_option> sensor_options[]{
    {"--dataset", sensor_option::dataset}, {"--sensor-grid-radius", sensor_option::radius},
    {"--sensor-grid-voxel-size", sensor_option::size}, {"--sensor-grid-factor", sensor_option::factor},
     {"--sensor-grid-truncation", sensor_option::truncation},
    {"--sensor-grid-frames", sensor_option::frames}, {"--sensor-grid-ray-cap", sensor_option::cap},
    {"--sensor-grid-first-entry", sensor_option::first}, {"--sensor-grid-backend", sensor_option::backend},
    {"--cuda-device", sensor_option::device}, {"--sensor-grid-gather", sensor_option::gather},
    {"--sensor-grid-short-list-max", sensor_option::threshold}
};
static_assert(valid_cli_options(sensor_options));
enum class sensor_backend { unknown, cpu, cuda };
inline constexpr cli_option_entry<sensor_backend> sensor_backends[]{
    {"cpu", sensor_backend::cpu}, {"cuda", sensor_backend::cuda}
};
static_assert(valid_cli_options(sensor_backends));
enum class sensor_boolean { unknown, enabled, disabled };
inline constexpr cli_option_entry<sensor_boolean> sensor_booleans[]{
    {"true", sensor_boolean::enabled}, {"false", sensor_boolean::disabled}
};
static_assert(valid_cli_options(sensor_booleans));
template<class number_type>
operation_status sensor_number(std::string_view text, number_type& value)
{
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);

    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        return std::unexpected(operation_error{"invalid sensor grid numeric argument"});
    return {};
}
inline int run_sensor_grid_cli(int argc, char** argv)
{
    const auto error = [](std::string_view message) {
        std::cerr << "covoxslam: " << message << '\n'; return 2;
    };
    sensor_grid::config settings;
    std::uint32_t factor{1};
    std::uint32_t frames{};
    sensor_grid::gather_policy policy;
    sensor_grid::gather_options gather;
    auto backend = sensor_backend::cpu;
    int device{};
     std::filesystem::path path;

    for (int index = 2; index < argc; ++index)
    {
        const auto option = lookup_cli_option(argv[index], sensor_options);

        if (option == sensor_option::unknown) return error("unknown sensor grid option");
        if (++index >= argc) return error("sensor grid option requires a value");
        const std::string_view value(argv[index]);
        operation_status status;
        switch (option) {
        case sensor_option::dataset: path = value; break;
        case sensor_option::radius: status = sensor_number(value, settings.radius); break;
        case sensor_option::size: status = sensor_number(value, settings.voxel_size); break;
         case sensor_option::factor: status = sensor_number(value, factor); break;
        case sensor_option::truncation: status = sensor_number(value, settings.truncation); break;
        case sensor_option::frames: status = sensor_number(value, frames); break;
        case sensor_option::cap: status = sensor_number(value, policy.max_rays); break;
        case sensor_option::device: status = sensor_number(value, device); break;
        case sensor_option::gather: status = select_sensor_gather(value, gather); break;
        case sensor_option::threshold: status = select_sensor_threshold(value, gather); break;
        case sensor_option::backend:
            backend = lookup_cli_option(value, sensor_backends);
            if (backend == sensor_backend::unknown) return error("sensor backend must be cpu or cuda");
            break;
        case sensor_option::first:
        {
            const auto selected = lookup_cli_option(value, sensor_booleans);

            if (selected == sensor_boolean::unknown) return error("first entry must be true or false");
            policy.first_entry_only = selected == sensor_boolean::enabled;
            break;
        }
        default: return error("unknown sensor grid option");
        }
        if (!status) return error(status.error().message);
    }
    const auto geometry = sensor_grid::make_geometry(settings);

    if (!geometry) return error(geometry.error().message);
    if (path.empty() || !factor) return error("sensor grid requires a dataset and positive factor");
    if (backend != sensor_backend::cuda && gather.backend != sensor_grid::gather_backend::csr)
        return error("prepared gathering requires the CUDA backend");
    if (!sensor_grid::valid_gather_policy(policy) || device < 0)
        return error("conflicting ray cap or invalid device");
    const auto input = load_dataset(path);
    if (frames > input.frames.size()) return error("requested sensor frame count exceeds dataset");
    const auto directions = sensor_grid::calibrated_directions(input.sensor);
    if (!directions) return error(directions.error().message);
    const auto start = std::chrono::steady_clock::now();
    const auto table = sensor_grid::build_lookup(settings, *directions, factor);

    if (!table) return error(table.error().message);
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "gpu_accumulation_bits=" << sensor_grid::accumulation_bits << " cpu_reference_bits=64\n";
    std::cout << "sensor_cells=" << table->grid.voxels << " active_cells=" << table->active_voxels.size()
              << " selected_rays=" << table->selected_rays << " intersections=" << table->intersections.size()
              << " lookup_setup_ms=" << elapsed << '\n';
    std::unique_ptr<sensor_grid::gpu_grid> gpu;
    const auto allocation_start = std::chrono::steady_clock::now();

    if (frames && backend == sensor_backend::cuda)
    {
        auto created = sensor_grid::gpu_grid::create(*table, device, 1073741824, gather);
        if (!created) return error(created.error().message);
         gpu = std::move(*created);
    }
    const auto allocation_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - allocation_start).count();
    double upload_ms{}, gather_ms{}, download_ms{};
    double prepare_ms{}, bin_ms[4]{};
    std::uint64_t observed{}, contributions{}, candidates{};
    std::vector<float> ranges(table->ray_count);

    for (std::uint32_t index = 0; index < frames; ++index)
    {
        const auto frame = load_frame(input, index, {false, false, true, true, false});

        if (frame.range_mm.size() != ranges.size() || frame.status.size() != ranges.size())
            return error("sensor frame dimensions differ from calibration");

        for (std::size_t ray = 0; ray < ranges.size(); ++ray)
            ranges[ray] = frame.status[ray] == 1 || frame.status[ray] == 3 ? frame.range_mm[ray] * 0.001F : 0.f;
        auto update_start = std::chrono::steady_clock::now();
        operation_result<std::vector<sensor_grid::voxel>> field;

        if (gpu) {
            const auto updated = gpu->update(ranges, policy);
            if (!updated) return error(updated.error().message);
            upload_ms += updated->upload_ms;
            gather_ms += updated->gather_ms;
            prepare_ms += updated->prepare_ms;

            for (unsigned bin = 0; bin < 4; ++bin) bin_ms[bin] += updated->bin_ms[bin];
            update_start = std::chrono::steady_clock::now();
            field = gpu->download();
            download_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - update_start).count();
        }
        else {
            field = sensor_grid::update_cpu(*table, ranges, policy);
            gather_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - update_start).count();
        }
        if (!field) return error(field.error().message);

        for (const auto value : *field) {
            observed += value.contributions != 0;
            contributions += value.contributions;
             candidates += value.candidates_tested;
        }
    }
    std::cout << "frames=" << frames << " observed_cells=" << observed << " contributions=" << contributions
              << " candidates_tested=" << candidates << " gpu_setup_ms=" << allocation_ms
              << " upload_ms=" << upload_ms << " gather_ms=" << gather_ms << " download_wall_ms=" << download_ms << '\n';
    if (gather.backend == sensor_grid::gather_backend::warp_soa_subwarp_prepared)
        std::cout << "prepare_ms=" << prepare_ms << " thread_ms=" << bin_ms[0] << " group8_ms=" << bin_ms[1]
                  << " group16_ms=" << bin_ms[2] << " warp_ms=" << bin_ms[3] << '\n';
    return 0;
}
}
