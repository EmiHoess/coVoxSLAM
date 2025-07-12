#include "cuda_resident_internal.hpp"
#include <bit>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace covoxslam
{
cuda_resident_context::cuda_resident_context() = default;
cuda_resident_context::~cuda_resident_context() = default;
std::size_t cuda_resident_context::device_bytes() const noexcept { return state->bytes; }
cuda_resident_order cuda_resident_context::fusion_order() const noexcept { return state->options.order; }
cuda_resident_reduction cuda_resident_context::reduction_mode() const noexcept { return state->options.reduction; }
bool cuda_resident_context::uses_step_tasks() const noexcept { return state->options.step_tasks; }
cuda_resident_warp_options cuda_resident_context::warp_configuration() const noexcept { return state->options.warp; }
cuda_free_space_options cuda_resident_context::free_space_configuration() const noexcept { return state->options.free_space; }
float cuda_resident_context::segment_length() const noexcept { return state->options.segment_length; }
bool cuda_resident_context::uses_compact() const noexcept { return state->options.compact; }
bool cuda_resident_context::uses_compact_batched() const noexcept { return state->options.compact_batched; }
bool cuda_resident_context::uses_fp32() const noexcept { return state->options.fp32; }



operation_result<tsdf_tiles> cuda_resident_context::download()
{
    const auto completed = flush();

    if (!completed) return std::unexpected(completed.error());
    auto& data = *state;
    data.failed = true;
    const auto count = completed->mapping.allocated_tiles;

    if (count > data.options.tile_capacity)
        return std::unexpected(operation_error{"resident tile count exceeds its capacity"});
    std::vector<detail::cuda_tile_coordinate> keys(static_cast<std::size_t>(count));
    std::vector<detail::cuda_tsdf_tile> tiles(keys.size());
    operation_status status;
    if (count) {
        status = data.check(cudaMemcpyAsync(keys.data(), data.view.keys, keys.size() * sizeof(keys[0]), cudaMemcpyDeviceToHost, data.stream));

        if (status) status = data.check(cudaMemcpyAsync(tiles.data(), data.view.tiles, tiles.size() * sizeof(tiles[0]), cudaMemcpyDeviceToHost, data.stream));
        ++data.statistics.explicit_waits;
        const auto drained = data.check(cudaStreamSynchronize(data.stream));

        if (status) status = drained;
    }

    if (!status) return std::unexpected(status.error());
    tsdf_tiles result;
    result.reserve(keys.size());

    for (std::size_t index = 0; index < keys.size(); ++index)
    {
        const auto key = keys[index];
        constexpr int bound = 1 << 20;

        if (key.x < -bound || key.x >= bound || key.y < -bound || key.y >= bound || key.z < -bound || key.z >= bound)
            return std::unexpected(operation_error{"resident tile coordinate is unsupported"});
        tsdf_tile tile;

        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& source = tiles[index].voxels[local];

            if (!std::isfinite(source.distance) || !std::isfinite(source.weight) || source.weight < 0 ||
                source.weight > data.config.maximum_weight || (source.flags & ~3U) ||
                std::abs(source.distance) > data.config.truncation_distance)
                return std::unexpected(operation_error{"invalid resident TSDF voxel"});
             tile.voxels[local] = {source.distance, source.weight, source.sample_count, source.flags};
        }
        if (!result.emplace(int3{key.x, key.y, key.z}, std::move(tile)).second)
            return std::unexpected(operation_error{"duplicate resident tile coordinate"});
    }
    data.statistics.mapping.cuda_downloaded_bytes += keys.size() * sizeof(keys[0]) + tiles.size() * sizeof(tiles[0]);
    data.failed = false;
    return result;
}

operation_status cuda_resident_context::reset()
{
    auto& data = *state;
    data.failed = true;
    auto status = data.check(cudaSetDevice(data.options.device));

    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (status) status = data.check(detail::cuda_initialize_resident_map(data.view, data.stream));

    if (status && data.free_space_flags)
        status = data.check(cudaMemsetAsync(data.free_space_flags, 0, std::size_t(data.options.tile_capacity) * 512, data.stream));
    if (status) status = data.initialize_compact();

    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (!status) return status;

    for (auto& slot : data.staging) slot.pending = false;
    data.next_slot = 0;
    data.next_scan = 0;
    data.statistics = {};
     data.failed = false;
    return {};
}

operation_status cuda_resident_context::restore(const tsdf_tiles& source)
{
    auto& data = *state;

    if (source.size() > data.options.tile_capacity)
        return std::unexpected(operation_error{"restored map exceeds resident tile capacity"});
    std::vector<int3> order;
    order.reserve(source.size());

    for (const auto& [key, tile] : source) order.push_back(key);
    std::sort(order.begin(), order.end(), block_index_less);
    std::vector<detail::cuda_tile_coordinate> keys;
    std::vector<detail::cuda_tsdf_tile> tiles(order.size());
    keys.reserve(order.size());

    for (std::size_t index = 0; index < order.size(); ++index)
    {
        const auto key = order[index];
        constexpr int bound = 1 << 20;

        if (key.x < -bound || key.x >= bound || key.y < -bound || key.y >= bound || key.z < -bound || key.z >= bound)
            return std::unexpected(operation_error{"restored tile coordinate is unsupported"});
        keys.push_back({key.x, key.y, key.z});
        const auto& input = source.find(key)->second;

        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& voxel = input.voxels[local];

            if (!std::isfinite(voxel.distance) || !std::isfinite(voxel.weight) || voxel.weight < 0 ||
                voxel.weight > data.config.maximum_weight || (voxel.flags & ~3U) ||
                std::abs(voxel.distance) > data.config.truncation_distance)
                return std::unexpected(operation_error{"invalid restored TSDF voxel"});
             tiles[index].voxels[local] = {voxel.distance, voxel.weight, voxel.sample_count, voxel.flags};
        }
    }
    auto status = reset();
    if (!status) return status;
    data.failed = true;

    if (!keys.empty()) {
        status = data.check(cudaMemcpyAsync(data.view.keys, keys.data(), keys.size() * sizeof(keys[0]), cudaMemcpyHostToDevice, data.stream));
        if (status) status = data.check(cudaMemcpyAsync(data.view.tiles, tiles.data(), tiles.size() * sizeof(tiles[0]), cudaMemcpyHostToDevice, data.stream));
    }

    if (status) status = data.check(detail::cuda_restore_resident_keys(data.view, static_cast<std::uint32_t>(keys.size()), data.stream));
    const auto drained = data.check(cudaStreamSynchronize(data.stream));

    if (!status) return status;
    if (!drained) return drained;
    data.failed = false;
    data.statistics.mapping.cuda_uploaded_bytes += keys.size() * sizeof(keys[0]) + tiles.size() * sizeof(tiles[0]);
    const auto completed = flush();

    if (!completed) return std::unexpected(completed.error());
    return {};
}
}
