#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/pgo.hpp>
#include <covoxslam/storage/schema.hpp>
#include <filesystem>

namespace covoxslam
{
enum class stored_pgo_fields { copied, borrowed, resident };
struct store_pipeline_options
{
    bool enabled{true};
    bool resident_mapping{true};
    bool negative_band{};
    esdf_backend esdf_generator{esdf_backend::cuda_block_band_soa};
    int device{};
    storage::layout layout{storage::layout::pool_soa};
    stored_pgo_fields fields{stored_pgo_fields::copied};
    std::filesystem::path checkpoint_root;
    std::size_t cpu_bytes{1073741824}, gpu_bytes{1073741824};
};
struct stored_pipeline_result
{
    std::string backend{"none"}, hash, publication{"none"}, report;
};
operation_status validate_store_pipeline(const store_pipeline_options& store, const pgo::options& solver);
operation_status validate_stored_export(std::span<const submap> maps, const store_pipeline_options& store);
}
