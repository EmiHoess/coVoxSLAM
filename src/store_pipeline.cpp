#include "store_pipeline_impl.hpp"

namespace covoxslam
{
operation_status validate_store_pipeline(const store_pipeline_options& store, const pgo::options& solver)
{
    if (!store.enabled) {
        if (store.fields != stored_pgo_fields::copied || !store.checkpoint_root.empty())
            return std::unexpected(operation_error{"store field and checkpoint options require a map store"});
        return {};
    }
    if (!store.cpu_bytes || !store.gpu_bytes)
        return std::unexpected(operation_error{"store memory budgets must be positive"});
    if (unsigned(store.layout) > unsigned(storage::layout::pool_soa) ||
        unsigned(store.fields) > unsigned(stored_pgo_fields::resident))
        return std::unexpected(operation_error{"invalid store layout or solver field selection"});
#if !defined(covoxslam_has_urd)
    if (store.layout != storage::layout::pooled_aos)
        return std::unexpected(operation_error{"SoA layouts require the optional collection dependency"});
#endif
#if !defined(covoxslam_has_cuda)
    if (store.fields == stored_pgo_fields::resident)
        return std::unexpected(operation_error{"resident PGO requires CUDA support"});
#endif
    if ((!solver.enabled && store.fields != stored_pgo_fields::copied) ||
        (store.fields == stored_pgo_fields::borrowed && solver.solver != pgo::backend::native_cpu) ||
        (store.fields == stored_pgo_fields::resident && solver.solver != pgo::backend::cuda_cold))
        return std::unexpected(operation_error{"solver and store field mode are incompatible"});
    if (!store.checkpoint_root.empty()) {
        std::error_code error;
        const bool exists = std::filesystem::exists(store.checkpoint_root, error);
        if (error || exists) return std::unexpected(operation_error{"checkpoint output must be a new directory"});
    }
    if (solver.enabled) return pgo::validate_options(solver);
    return {};
}
operation_result<stored_pipeline_result> finish_stored_pipeline(std::span<submap> maps,
    const store_pipeline_options& store, const pgo::options& solver, bool shadow)
{
    auto valid = validate_store_pipeline(store, solver);
    if (!valid) return std::unexpected(valid.error());
    if (!store.enabled || maps.empty()) return std::unexpected(operation_error{"stored pipeline requires completed maps"});
    const auto& first = maps.front().config;
    for (const auto& map : maps)
        if (!map.config.valid() || map.config.voxel_size != first.voxel_size ||
            map.config.esdf_maximum_distance_m != first.esdf_maximum_distance_m)
            return std::unexpected(operation_error{"stored submap scalar field parameters differ"});

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
