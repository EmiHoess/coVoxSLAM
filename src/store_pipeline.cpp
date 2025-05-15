#include "store_pipeline_impl.hpp"

namespace covoxslam
{
operation_status validate_store_pipeline(const store_pipeline_options& store, const pgo::options& solver)
{
    if (!store.enabled) return {};
    if (!store.cpu_bytes || !store.gpu_bytes)
        return std::unexpected(operation_error{"store memory budgets must be positive"});
    if (solver.enabled) return pgo::validate_options(solver);
    return {};
}
operation_result<stored_pipeline_result> finish_stored_pipeline(std::span<submap> maps,
    const store_pipeline_options& store, const pgo::options& solver, bool shadow)
{
    auto valid = validate_store_pipeline(store, solver);
    if (!valid) return std::unexpected(valid.error());
    if (!store.enabled || maps.empty()) return std::unexpected(operation_error{"stored pipeline requires completed maps"});

    switch (store.layout) {
    case storage::layout::pooled_aos: return detail::stored_pipeline<storage::layout::pooled_aos>(maps, store, solver, shadow);
#if defined(covoxslam_has_urd)
    case storage::layout::pool_soa: return detail::stored_pipeline<storage::layout::pool_soa>(maps, store, solver, shadow);
    case storage::layout::tile_aosoa: return detail::stored_pipeline<storage::layout::tile_aosoa>(maps, store, solver, shadow);
#endif
    default: return std::unexpected(operation_error{"requested store layout was not built"});
    }
}
}
