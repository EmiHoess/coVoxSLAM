#include <covoxslam/resident_pipeline.hpp>
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
#include "store_pipeline_impl.hpp"
#endif

namespace covoxslam
{
struct resident_pipeline::implementation
{
    store_pipeline_options options;
    resident_pipeline_metrics metrics;
    bool failed{}, finished{};
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
    std::unique_ptr<store::coordinated_store<storage::layout::pool_soa>> maps;
#endif
};
resident_pipeline::resident_pipeline() = default;
resident_pipeline::~resident_pipeline() = default;
resident_pipeline_metrics resident_pipeline::statistics() const { return state_->metrics; }
operation_result<std::unique_ptr<resident_pipeline>> resident_pipeline::create(
    const store_pipeline_options& options, const pgo::options& solver)
{
    auto checked = validate_store_pipeline(options, solver);
    if (!checked) return std::unexpected(checked.error());

    if (!options.enabled || options.layout != storage::layout::pool_soa ||
        options.esdf_generator != esdf_backend::cuda_block_band_soa)
        return std::unexpected(operation_error{"resident mapping requires SoA storage and SoA band ESDF"});
#if defined(covoxslam_has_cuda) && defined(covoxslam_has_urd)
    checked = storage::cuda_storage_status(cudaSetDevice(options.device));
    if (!checked) return std::unexpected(checked.error());
    auto owner = std::unique_ptr<resident_pipeline>(new resident_pipeline);
    owner->state_ = std::make_unique<implementation>();
    owner->state_->options = options;
    store::residency_policy policy;
    policy.limits = {options.cpu_bytes, 128U * 1024U * 1024U, options.gpu_bytes};
     owner->state_->maps = std::make_unique<store::coordinated_store<storage::layout::pool_soa>>(policy);
    return owner;
#else
    return std::unexpected(operation_error{"resident mapping requires CUDA and collection support"});
#endif
}
}
