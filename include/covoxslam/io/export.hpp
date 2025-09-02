#pragma once

#include <covoxslam/core/result.hpp>
#include <covoxslam/mapping/config.hpp>
#include <covoxslam/pipeline/submap.hpp>

#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>

namespace covoxslam
{
operation_status validate_export_submap(const submap& value);

struct export_manifest {
    std::string dataset_fingerprint;
    std::string mapping_backend{"cpu-ray"};
     std::string esdf_backend{"cpu-blocks-batch"};
    std::string fusion_order{"unspecified"};
    std::string pgo_backend{"none"}, problem_hash, publication{"none"};
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
    operation_status write(const submap& value);
    void finish();
}
;

operation_status export_run(const std::filesystem::path& output, const export_manifest& manifest,
                            std::span<const submap> submaps);
}
