#include "sensor_grid_map_internal.hpp"
#if defined(covoxslam_has_urd)
#include <covoxslam/store/coordinator.hpp>
#endif

namespace covoxslam::sensor_grid
{
operation_result<std::size_t> gpu_map::retain(
    store::coordinated_store<storage::layout::pool_soa, storage::schema::esdf>& destination,
    std::uint64_t id, pose4 anchor)
{
#if defined(covoxslam_has_urd)
    auto& data = *state;

    if (data.failed || !data.soa) return std::unexpected(operation_error{"resident retention requires a healthy SoA mapper"});
    auto checked = data.check(cudaSetDevice(data.device));
    if (checked) checked = data.wait(checked);
    if (!checked) return std::unexpected(checked.error());
    if (data.indices.size() > data.capacity || data.indices.size() != data.index.size())
        return std::unexpected(operation_error{"resident mapper topology is inconsistent"});
    std::vector<store::block_key> keys;
    keys.reserve(data.indices.size());

    for (std::size_t slot = 0; slot < data.indices.size(); ++slot)
    {
        const auto key = data.indices[slot];
        const auto found = data.index.find(key);

        if (found == data.index.end() || found->second != slot)
            return std::unexpected(operation_error{"resident mapper slot ordering is invalid"});
        keys.push_back({id, key.x, key.y, key.z});
    }
    checked = destination.import_device_tsdf(id, {anchor.x, anchor.y, anchor.z, anchor.yaw}, keys, *data.soa);
    if (!checked) return std::unexpected(checked.error());
    return keys.size();
#else
    return std::unexpected(operation_error{"resident retention requires collection support"});
#endif
}
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
