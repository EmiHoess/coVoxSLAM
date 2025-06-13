#include "sensor_grid_map_internal.hpp"

namespace covoxslam::sensor_grid
{
operation_result<tsdf_tiles> gpu_map::implementation::download_soa()
{
#if defined(covoxslam_has_urd)
    if (!soa || failed) return std::unexpected(operation_error{"SoA mapping payload is unavailable"});
    auto checked = check(cudaSetDevice(device));
    if (checked) checked = wait(checked);
    if (!checked) return std::unexpected(checked.error());
    storage::draft<storage::schema::tsdf, storage::layout::pool_soa> staging;
    checked = staging.initialize(indices.size());
    if (!checked) return std::unexpected(checked.error());
    storage::transfer_measurement measured;
    checked = soa->download_range(staging, 0, 0, indices.size() * 512, 15,
        storage::transfer_method::bulk_same_layout, measured);

    if (!checked) return std::unexpected(checked.error());
    tsdf_tiles output;
    output.reserve(indices.size());
    const auto view = staging.view();

    for (std::size_t slot = 0; slot < indices.size(); ++slot)
    {
        auto& destination = output[indices[slot]];

        for (std::size_t local = 0; local < 512; ++local) {
            const auto value = view[slot].voxels[local];
             destination.voxels[local] = {value.distance, value.weight, value.sample_count, value.flags};
        }
    }
    return output;
#else
    return std::unexpected(operation_error{"SoA mapping requires collection support"});
#endif
}
}
