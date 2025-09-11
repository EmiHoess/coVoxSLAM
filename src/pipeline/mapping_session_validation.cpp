#include "mapping_session_internal.hpp"

namespace covoxslam
{
operation_status validate_stored_export(std::span<const submap> maps, const store_pipeline_options& store)
{
    for (const auto& map : maps) {
        auto identity = detail::stored_field_parameters(map, store);
        if (!identity) return std::unexpected(identity.error());

        for (const auto& [key, tile] : map.tsdf)

            if (!map.esdf.contains(key))
                return std::unexpected(operation_error{"generated ESDF export topology differs from TSDF"});
    }
    return {};
}
operation_status validate_store_pipeline(const store_pipeline_options& store, const pgo::options& solver)
{
    if (!store.enabled || !store.resident_mapping || store.layout != storage::layout::pool_soa ||
        store.esdf_generator != esdf_backend::cuda_block_band_soa || store.device < 0)
        return std::unexpected(operation_error{"coVoxSLAM requires resident SoA storage and band ESDF"});
#if !defined(covoxslam_has_cuda) || !defined(covoxslam_has_urd)
    return std::unexpected(operation_error{"coVoxSLAM mapping requires CUDA and collection support"});
#endif
    if (!store.cpu_bytes || !store.gpu_bytes)
        return std::unexpected(operation_error{"store memory budgets must be positive"});
    if (unsigned(store.fields) > unsigned(stored_pgo_fields::resident))
        return std::unexpected(operation_error{"invalid solver field selection"});
    if (solver.enabled && store.fields == stored_pgo_fields::resident && solver.device != store.device)
        return std::unexpected(operation_error{"resident solver and ESDF must use the same CUDA device"});
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
}
