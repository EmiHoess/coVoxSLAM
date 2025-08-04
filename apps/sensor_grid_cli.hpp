#pragma once
#include <covoxslam/sensor_grid_calibration.hpp>
#include <covoxslam/sensor_grid.hpp>
#include "retired_options.hpp"
#include <iostream>

namespace covoxslam
{
inline int run_sensor_grid_cli(int argc, char** argv)
{
    const auto error = [](std::string_view message) {
        std::cerr << "covoxslam: " << message << '\n';
        return 2;
    };
    std::filesystem::path path;

    for (int index = 2; index < argc; index += 2)
    {
        const std::string_view option(argv[index]);

        if (index + 1 >= argc) return error("sensor grid option requires a value");
        const auto active = validate_active_option(option);

        if (!active) return error(active.error().message);
        if (option != "--dataset") return error("option is not applicable to coVoxSLAM lookup inspection");
         path = argv[index + 1];
    }

    if (path.empty()) return error("sensor grid requires a dataset");
    const auto input = load_dataset(path);

    if (!input) return error(input.error().message);
    const auto directions = sensor_grid::calibrated_directions(input->sensor);
    if (!directions) return error(directions.error().message);
    auto table = sensor_grid::build_lookup({}, *directions);

    if (!table) return error(table.error().message);
    std::cout << "coVoxSLAM lookup, cells " << table->grid.voxels
        << ", active cells " << table->active_voxels.size()
        << ", rays " << table->ray_count
        << ", intersections " << table->intersections.size() << '\n';
    return 0;
}
}
