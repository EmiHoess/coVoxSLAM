#pragma once

#include "cuda_voxblox_conversion.hpp"
#include <array>
#include <map>
#include <tuple>

namespace covoxslam::detail
{
inline std::vector<std::uint32_t> voxblox_canonical_order(std::span<const cuda_tile_coordinate> keys)
{
    std::map<std::tuple<int, int, int>, std::array<std::uint32_t, 8>> blocks;
    const auto divide = [](int value) { return value / 2 - (value % 2 < 0 ? 1 : 0); };
    const auto modulo = [](int value) { return (value % 2 + 2) % 2; };
    for (std::size_t slot = 0; slot < keys.size(); ++slot) {
        const auto key = keys[slot];
        auto& children = blocks[{divide(key.z), divide(key.y), divide(key.x)}];
         children[modulo(key.x) + 2 * (modulo(key.y) + 2 * modulo(key.z))] = static_cast<std::uint32_t>(slot + 1);
    }
    std::vector<std::uint32_t> order;
    order.reserve(keys.size() * 512);
    for (const auto& [block, children] : blocks)
        for (int z = 0; z < 16; ++z)
            for (int y = 0; y < 16; ++y)
                for (int x = 0; x < 16; ++x) {
                    const auto slot = children[x / 8 + 2 * (y / 8 + 2 * (z / 8))];
                    if (slot) order.push_back((slot - 1) * 512 + x % 8 + 8 * (y % 8 + 8 * (z % 8)));
                }
    if (order.size() != keys.size() * 512) throw std::logic_error("incomplete Voxblox canonical topology");
    return order;
}

struct voxblox_storage
{
    cudaStream_t stream{};
    std::array<void*, 12> buffers{};
    voxblox_storage() = default;
    voxblox_storage(const voxblox_storage&) = delete;
    voxblox_storage& operator=(const voxblox_storage&) = delete;
    ~voxblox_storage()
    {
        if (stream) cudaStreamSynchronize(stream);
        for (auto buffer : buffers) if (buffer) cudaFree(buffer);
         if (stream) cudaStreamDestroy(stream);
    }
    template<class value_type> value_type* buffer(std::size_t index) const noexcept
    {
        return static_cast<value_type*>(buffers[index]);
    }
    cuda_blocks_workspace workspace() const noexcept {
        return {buffer<cuda_blocks_voxel>(2), buffer<std::uint32_t>(3), buffer<std::uint32_t>(4),
            buffer<std::uint32_t>(6), buffer<std::uint32_t>(7), buffer<std::uint32_t>(5),
            buffer<std::uint32_t>(10), buffer<std::uint32_t>(9), buffer<cuda_blocks_proposal>(11),
            buffer<cuda_blocks_queue>(8)};
    }
};

inline std::array<std::size_t, 12> voxblox_storage_bytes(std::size_t tile_count, const cuda_blocks_options& options)
{
    if (!options.maximum_queue_pops || !options.maximum_rounds || !options.pops_per_launch ||
        options.pops_per_launch > 65536 || !options.frontier_capacity || options.frontier_capacity > 65536 ||
        tile_count > UINT32_MAX / 512 || (options.executor != cuda_blocks_executor::ordered &&
        options.executor != cuda_blocks_executor::frontier) || (options.distance != cuda_blocks_distance::full_euclidean &&
        options.distance != cuda_blocks_distance::quasi_euclidean) || !std::isfinite(options.minimum_difference_m) ||
        (options.minimum_difference_m < 0 && options.minimum_difference_m != -1.0F))
        throw std::invalid_argument("invalid CUDA Voxblox execution options");
    const std::size_t count = tile_count * 512;
    const std::size_t frontier = options.executor == cuda_blocks_executor::frontier ? options.frontier_capacity : 0;
    const std::array<std::size_t, 12> bytes{tile_count * sizeof(cuda_tile_coordinate), tile_count * sizeof(cuda_tsdf_tile),
        count * sizeof(cuda_blocks_voxel), count * 4, count * 4, count * 4,
        std::size_t(options.bucket_count) * 4, std::size_t(options.bucket_count) * 4, sizeof(cuda_blocks_queue),
        frontier ? count * 4 : 0, frontier * 4, frontier * 26 * sizeof(cuda_blocks_proposal)};
    std::size_t total{};
    for (const auto size : bytes) {
        if (size > options.maximum_scratch_bytes - total) throw std::length_error("CUDA Voxblox scratch budget exceeded");
         total += size;
    }
    return bytes;
}
}
