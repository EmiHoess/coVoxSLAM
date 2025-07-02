#pragma once
#include <covoxslam/store/esdf_resources.hpp>
#include "../../../cuda/store/mapping.h"

namespace covoxslam::store
{
struct esdf_execution_limits
{
    std::size_t scratch_bytes{256U * 1024U * 1024U};
    std::uint32_t maximum_rounds{10000}, pops_per_launch{4096};
    std::uint64_t maximum_pops{100000000};
};
template<storage::layout l> operation_status generate_site_esdf(
    coordinated_store<l, storage::schema::projective_esdf>& maps, std::uint64_t id,
    detail::cuda_site_esdf_parameters parameters, esdf_execution_limits limits = {})
{
    auto validated = esdf_launch_status(detail::cuda_initialize_site_esdf(nullptr, nullptr, 0, parameters, nullptr, nullptr, nullptr));
    if (!validated) return validated;
    if (!limits.maximum_rounds) return std::unexpected(operation_error{"ESDF iteration limit is zero"});
    return maps.derive_gpu(id, [&](auto source, auto source_index, auto target, auto target_index, auto stream,
        std::span<const block_key> keys, memory_budget& budget) -> operation_status {
        const auto count = keys.size();
        if (!count || count > UINT32_MAX / 512) return std::unexpected(operation_error{"ESDF topology is empty or oversized"});
        const std::array<std::size_t, 5> sizes{keys.size_bytes(), count * sizeof(detail::cuda_tile_coordinate),
            count * sizeof(detail::cuda_tsdf_tile), count * 512 * sizeof(detail::cuda_site_esdf_voxel), 8};
        esdf_scratch scratch;
        auto checked = scratch.initialize(sizes, limits.scratch_bytes, budget, stream);
        if (!checked) return checked;
        auto* device_keys = scratch.get<block_key>(0);
        auto* ordered_keys = scratch.get<detail::cuda_tile_coordinate>(1);
        auto* input = scratch.get<detail::cuda_tsdf_tile>(2);
        auto* output = scratch.get<detail::cuda_site_esdf_voxel>(3);
        auto* flags = scratch.get<unsigned>(4);
        checked = storage::cuda_storage_status(cudaMemcpyAsync(device_keys, keys.data(), keys.size_bytes(), cudaMemcpyHostToDevice, stream));
        if (!checked) return checked;
        checked = esdf_launch_status(pack_esdf_input(source.native_accessor().descriptor, source_index, device_keys,
            unsigned(count), ordered_keys, input, flags, stream));
        if (!checked) return checked;
        checked = scratch.check(flags);
        if (!checked) return checked;
        checked = esdf_launch_status(detail::cuda_initialize_site_esdf(ordered_keys, input, count, parameters, output, flags, stream));
        if (!checked) return checked;
        checked = scratch.check(flags);
        if (!checked) return checked;
        const float distance = parameters.maximum_distance_m / parameters.voxel_size;
        unsigned changed{1}, rounds{};
        while (changed && rounds < limits.maximum_rounds)
        {
            checked = storage::cuda_storage_status(cudaMemsetAsync(flags + 1, 0, 4, stream));
            if (!checked) return checked;
            checked = esdf_launch_status(detail::cuda_sweep_site_esdf(ordered_keys, count, output, distance * distance, flags + 1, stream));
            if (!checked) return checked;
            checked = esdf_launch_status(detail::cuda_relax_site_esdf_neighbors(ordered_keys, count, output, distance * distance, flags + 1, stream));
            if (!checked) return checked;
            checked = scratch.read(changed, flags + 1);
            if (!checked) return checked;
            ++rounds;
        }
        if (changed) return std::unexpected(operation_error{"ESDF iteration limit exceeded"});
        checked = esdf_launch_status(unpack_esdf_output(target.native_accessor().descriptor, target_index, device_keys,
            unsigned(count), output, storage::schema::projective_esdf, flags, stream));
        if (!checked) return checked;
        return scratch.check(flags);
    });
}
}
