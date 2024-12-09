#pragma once

#include <covoxslam/sensor_grid_calibration.hpp>
#include "cli_lookup.hpp"
#include <charconv>
#include <chrono>
#include <iostream>

namespace covoxslam
{
enum class sensor_option { unknown, dataset, radius, size, factor, truncation };
inline constexpr cli_option_entry<sensor_option> sensor_options[]{
    {"--dataset", sensor_option::dataset}, {"--sensor-grid-radius", sensor_option::radius},
    {"--sensor-grid-voxel-size", sensor_option::size}, {"--sensor-grid-factor", sensor_option::factor},
     {"--sensor-grid-truncation", sensor_option::truncation}
};
static_assert(valid_cli_options(sensor_options));
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
        default: return error("unknown sensor grid option");
        }
        if (!status) return error(status.error().message);
    }
    const auto geometry = sensor_grid::make_geometry(settings);

    if (!geometry) return error(geometry.error().message);
    if (path.empty() || !factor) return error("sensor grid requires a dataset and positive factor");
    const auto input = load_dataset(path);
    const auto directions = sensor_grid::calibrated_directions(input.sensor);
    if (!directions) return error(directions.error().message);
    const auto start = std::chrono::steady_clock::now();
    const auto table = sensor_grid::build_lookup(settings, *directions, factor);

    if (!table) return error(table.error().message);
    const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "sensor_cells=" << table->grid.voxels << " active_cells=" << table->active_voxels.size()
              << " selected_rays=" << table->selected_rays << " intersections=" << table->intersections.size()
              << " lookup_setup_ms=" << elapsed << '\n';
    return 0;
}
}
