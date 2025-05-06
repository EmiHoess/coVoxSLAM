#pragma once
#include <covoxslam/store/gpu_esdf.hpp>
#include <numeric>
#include <tuple>

namespace covoxslam::store
{
inline std::vector<std::uint32_t> esdf_seed_order(std::span<const block_key> keys)
{
    std::vector<std::uint32_t> order(keys.size() * 512);
    std::iota(order.begin(), order.end(), 0U);
    const auto rank = [&](std::uint32_t i) {
        const auto key = keys[i / 512];
        const auto local = i % 512;
        const std::int64_t x = std::int64_t(key.x) * 8 + (local & 7);
        const std::int64_t y = std::int64_t(key.y) * 8 + ((local >> 3) & 7);
        const std::int64_t z = std::int64_t(key.z) * 8 + (local >> 6);
        const auto floor = [](std::int64_t v) { return v / 16 - (v % 16 < 0); };
        return std::tuple{floor(z), floor(y), floor(x), (z % 16 + 16) % 16,
            (y % 16 + 16) % 16, (x % 16 + 16) % 16};
    };
    std::sort(order.begin(), order.end(), [&](auto a, auto b) { return rank(a) < rank(b); });
    return order;
}
template<storage::layout l> operation_status generate_blocks_esdf(
    coordinated_store<l, storage::schema::blocks_esdf>& maps, std::uint64_t id,
    detail::cuda_blocks_parameters parameters, esdf_execution_limits limits = {})
{
    auto validated = esdf_launch_status(detail::cuda_initialize_blocks(nullptr, nullptr, 0, parameters, {}, nullptr));
    if (!validated) return validated;
    if (!limits.maximum_rounds || !limits.maximum_pops || !limits.pops_per_launch || limits.pops_per_launch > 65536)
        return std::unexpected(operation_error{"ESDF propagation limits are invalid"});
    return maps.derive_gpu(id, [&](auto source, auto source_index, auto target, auto target_index, auto stream,
        std::span<const block_key> keys, memory_budget& budget) -> operation_status {
        const auto count = keys.size();
        if (!count || count > UINT32_MAX / 512) return std::unexpected(operation_error{"ESDF topology is empty or oversized"});
        const auto voxels = count * 512;
        const std::array<std::size_t, 11> sizes{keys.size_bytes(), count * sizeof(detail::cuda_tile_coordinate),
            count * sizeof(detail::cuda_tsdf_tile), voxels * sizeof(detail::cuda_blocks_voxel),
            voxels * 4, voxels * 4, voxels * 4, parameters.bucket_count * std::size_t(4),
            parameters.bucket_count * std::size_t(4), sizeof(detail::cuda_blocks_queue), 4};
        auto host_memory = budget.reserve({voxels * 4, 0, 0});
        if (!host_memory) return std::unexpected(host_memory.error());
        esdf_scratch scratch;
        auto checked = scratch.initialize(sizes, limits.scratch_bytes, budget, stream);
        if (!checked) return checked;
        const auto order = esdf_seed_order(keys);
        auto* device_keys = scratch.get<block_key>(0);
        auto* ordered_keys = scratch.get<detail::cuda_tile_coordinate>(1);
        auto* input = scratch.get<detail::cuda_tsdf_tile>(2);
        auto* flags = scratch.get<unsigned>(10);
        const detail::cuda_blocks_workspace work{scratch.get<detail::cuda_blocks_voxel>(3),
            scratch.get<unsigned>(4), scratch.get<unsigned>(5), scratch.get<unsigned>(7), scratch.get<unsigned>(8),
            scratch.get<unsigned>(6), nullptr, nullptr, nullptr, scratch.get<detail::cuda_blocks_queue>(9)};
        checked = storage::cuda_storage_status(cudaMemcpyAsync(device_keys, keys.data(), keys.size_bytes(), cudaMemcpyHostToDevice, stream));
        if (!checked) return checked;
        checked = storage::cuda_storage_status(cudaMemcpyAsync(scratch.buffers[6], order.data(), voxels * 4, cudaMemcpyHostToDevice, stream));
        if (!checked) return checked;
        checked = esdf_launch_status(pack_esdf_input(source.native_accessor().descriptor, source_index, device_keys,
            unsigned(count), ordered_keys, input, flags, stream));
        if (!checked) return checked;
        checked = scratch.check(flags);
        if (!checked) return checked;
        detail::cuda_blocks_queue state;
        auto read_state = [&]() -> operation_status {
            auto copied = scratch.read(state, work.queue);
            if (!copied) return copied;
            if (state.status) return std::unexpected(operation_error{"ESDF queue validation failed"});
            return {};
        };
        checked = esdf_launch_status(detail::cuda_initialize_blocks(ordered_keys, input, count, parameters, work, stream));
        if (!checked) return checked;
        checked = read_state();
        if (!checked) return checked;
        checked = esdf_launch_status(detail::cuda_enqueue_blocks_seeds(voxels, parameters, work, stream));
        if (!checked) return checked;
        checked = read_state();
        if (!checked) return checked;
        unsigned rounds{};

        while (state.size) {
            if (rounds >= limits.maximum_rounds || state.pops >= limits.maximum_pops)
                return std::unexpected(operation_error{"ESDF propagation limit exceeded"});
            const auto pops = unsigned(std::min<std::uint64_t>(limits.maximum_pops - state.pops, limits.pops_per_launch));
            checked = esdf_launch_status(detail::cuda_propagate_blocks_ordered(ordered_keys, count, parameters, work, pops, stream));
            if (!checked) return checked;
            checked = read_state();
            if (!checked) return checked;
            ++rounds;
        }
        checked = esdf_launch_status(detail::cuda_validate_blocks(ordered_keys, input, count, parameters, work, stream));
        if (!checked) return checked;
        checked = read_state();
        if (!checked) return checked;
        checked = esdf_launch_status(unpack_esdf_output(target.native_accessor().descriptor, target_index, device_keys,
            unsigned(count), work.values, storage::schema::blocks_esdf, flags, stream));
        if (!checked) return checked;
        return scratch.check(flags);
    });
}
}
