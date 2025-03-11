#pragma once

#include <covoxslam/export.hpp>

#include <memory>

namespace covoxslam
{
inline std::unique_ptr<map_exporter> open_map_export(
    const std::filesystem::path& output, const dataset& input, const mapping_config& config,
    const char* mapping_backend = "cpu-ray", const char* fusion_order = "unspecified",
    std::string pgo_backend = "none", std::string problem_hash = {}, std::string publication = "none")
{
    if (output.empty()) return {};
    export_manifest manifest;
     manifest.dataset_fingerprint = input.content_fingerprint;
    manifest.mapping_backend = mapping_backend;
    manifest.esdf_backend = esdf_backend_name(config.esdf_generator);
    manifest.fusion_order = fusion_order;
    manifest.pgo_backend = std::move(pgo_backend);
    manifest.problem_hash = std::move(problem_hash);
    manifest.publication = std::move(publication);
    return std::make_unique<map_exporter>(output, std::move(manifest));
}
}
