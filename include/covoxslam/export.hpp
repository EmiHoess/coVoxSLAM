#pragma once

#include <covoxslam/mapping.hpp>

#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>

namespace covoxslam
{
struct export_manifest {
    std::string dataset_fingerprint;
    std::string mapping_backend{"cpu-ray"};
     std::string esdf_backend{"cpu-voxblox-batch"};
};

class map_exporter
{
    std::filesystem::path directory;
    export_manifest manifest;
    std::ofstream poses;
    std::ofstream metadata;
    std::optional<mapping_config> initial_config;
     std::optional<std::size_t> previous_id;
    std::size_t count{};
    bool failed{};
    bool finished{};
public:
    map_exporter(const std::filesystem::path& output, export_manifest description);
    map_exporter(const map_exporter&) = delete;
    map_exporter& operator=(const map_exporter&) = delete;
    void write(const submap& value);
    void finish();
}
;

void export_run(const std::filesystem::path& output, const export_manifest& manifest,
                 std::span<const submap> submaps);
}
