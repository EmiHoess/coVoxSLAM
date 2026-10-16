#pragma once

#include <covoxslam/io/export.hpp>

#include <memory>

namespace covoxslam
{
inline operation_result<std::unique_ptr<map_exporter>> open_map_export(
    const std::filesystem::path& output, const dataset& input, const mapping_config& config,
    const char* mapping_backend = "cpu-ray", const char* fusion_order = "unspecified",
    std::string pgo_backend = "none", std::string problem_hash = {}, std::string publication = "none")
{
    if (output.empty()) return std::unique_ptr<map_exporter>{};
    export_manifest manifest;
     manifest.dataset_fingerprint = input.content_fingerprint;
    manifest.mapping_backend = mapping_backend;
    manifest.esdf_backend = esdf_backend_name(config.esdf_generator);
    manifest.fusion_order = fusion_order;
    manifest.pgo_backend = std::move(pgo_backend);
    manifest.problem_hash = std::move(problem_hash);
    manifest.publication = std::move(publication);
    return map_exporter::create(output, std::move(manifest));
}
}
