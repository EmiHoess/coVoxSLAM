#include <covoxslam/cuda.hpp>
#include "sparse_mapping.h"
#include "cuda_site_conversion.hpp"
#include "cuda_blocks_conversion.hpp"

#include <cuda_runtime_api.h>
#include "cuda_blocks_storage.hpp"
#include <array>
#include <algorithm>
#include <vector>
#include <limits>
#include <stdexcept>

#include <type_traits>


namespace covoxslam
{
namespace
{
operation_status cuda_status(cudaError_t error)
{
    if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
    return {};
}

void check_cuda(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}


}


#include <algorithm>
#include <chrono>
#include <cmath>



namespace covoxslam
{
static_assert(sizeof(detail::cuda_tile_coordinate) == 12);
static_assert(sizeof(detail::cuda_tsdf_tile) == 8192);
static_assert(sizeof(detail::cuda_site_esdf_voxel) == 20);
namespace
{
struct site_esdf_storage
{
    cudaStream_t stream{};
    std::array<void*, 4> buffers{};
    ~site_esdf_storage()
    {
        if (stream) cudaStreamSynchronize(stream);
        for (auto buffer : buffers) if (buffer) cudaFree(buffer);
         if (stream) cudaStreamDestroy(stream);
    }
    template<class value_type> value_type* buffer(std::size_t index) const noexcept
    {
        return static_cast<value_type*>(buffers[index]);
    }
};
}

cuda_site_esdf_statistics build_cuda_site_esdf(const tsdf_tiles& tsdf, esdf_tiles& esdf,
    const mapping_config& config, const cuda_site_esdf_options& options)
{
    const auto start = std::chrono::steady_clock::now();
    const detail::cuda_site_esdf_parameters parameters{config.voxel_size, config.truncation_distance,
        config.esdf_site_band_m, config.esdf_minimum_weight, config.esdf_maximum_distance_m,
        static_cast<detail::cuda_esdf_unobserved>(options.unobserved), options.include_negative_truncation_band};
    check_cuda(static_cast<cudaError_t>(detail::cuda_initialize_site_esdf(
        nullptr, nullptr, 0, parameters, nullptr, nullptr, nullptr)));
    constexpr std::size_t bytes_per_tile = sizeof(detail::cuda_tile_coordinate) +
        sizeof(detail::cuda_tsdf_tile) + tile_voxels * sizeof(detail::cuda_site_esdf_voxel);
    if (!options.maximum_iterations || !config.maximum_tiles || tsdf.size() > config.maximum_tiles ||
        tsdf.size() > (std::numeric_limits<unsigned>::max() / tile_voxels) ||
        options.maximum_scratch_bytes < 8 || tsdf.size() > (options.maximum_scratch_bytes - 8) / bytes_per_tile)
        throw std::length_error("CUDA site ESDF exceeds its scratch or iteration limits");
    const auto device = query_cuda_device(options.device);
    if (tsdf.size() * bytes_per_tile + 8 > device.total_memory_bytes)
        throw std::length_error("CUDA site ESDF exceeds device memory");
    check_cuda(cudaSetDevice(options.device));
    cuda_site_esdf_statistics statistics;
    if (tsdf.empty()) { esdf.clear(); return statistics; }
    std::vector<int3> order;
    order.reserve(tsdf.size());
    for (const auto& [key, tile] : tsdf) order.push_back(key);
    std::sort(order.begin(), order.end(), block_index_less);
     std::vector<detail::cuda_tile_coordinate> keys;
    std::vector<detail::cuda_tsdf_tile> tiles(tsdf.size());
    std::vector<detail::cuda_site_esdf_voxel> values(tsdf.size() * tile_voxels);
    keys.reserve(tsdf.size());
    for (std::size_t slot = 0; slot < order.size(); ++slot)
    {
        const auto key = order[slot];
        keys.push_back({key.x, key.y, key.z});
        const auto& source = tsdf.at(key);
        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& voxel = source.voxels[local];
            tiles[slot].voxels[local] = {voxel.distance, voxel.weight, voxel.sample_count, voxel.flags};
        }
    }
    site_esdf_storage storage;
    check_cuda(cudaStreamCreateWithFlags(&storage.stream, cudaStreamNonBlocking));
    const std::array<std::size_t, 4> bytes{keys.size() * sizeof(keys[0]), tiles.size() * sizeof(tiles[0]),
        values.size() * sizeof(values[0]), 2 * sizeof(std::uint32_t)};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        check_cuda(cudaMalloc(&storage.buffers[index], bytes[index]));
    auto* device_keys = storage.buffer<detail::cuda_tile_coordinate>(0);
    auto* device_tiles = storage.buffer<detail::cuda_tsdf_tile>(1);
    auto* device_values = storage.buffer<detail::cuda_site_esdf_voxel>(2);
     auto* flags = storage.buffer<std::uint32_t>(3);
    check_cuda(cudaMemcpyAsync(device_keys, keys.data(), bytes[0], cudaMemcpyHostToDevice, storage.stream));
    check_cuda(cudaMemcpyAsync(device_tiles, tiles.data(), bytes[1], cudaMemcpyHostToDevice, storage.stream));
    check_cuda(cudaMemsetAsync(flags, 0, bytes[3], storage.stream));
    statistics.uploaded_bytes = bytes[0] + bytes[1];
    check_cuda(static_cast<cudaError_t>(detail::cuda_initialize_site_esdf(
        device_keys, device_tiles, tsdf.size(), parameters, device_values, flags, storage.stream)));
    std::uint32_t status{};
    check_cuda(cudaMemcpyAsync(&status, flags, sizeof(status), cudaMemcpyDeviceToHost, storage.stream));
    check_cuda(cudaStreamSynchronize(storage.stream));
    statistics.downloaded_bytes += sizeof(status);
    if (status) throw std::invalid_argument("invalid CUDA site ESDF input tiles");
    const float maximum_voxels = parameters.maximum_distance_m / parameters.voxel_size;
    const float maximum_squared = maximum_voxels * maximum_voxels;
    std::uint32_t changed = 1;
    while (changed && statistics.wave_iterations < options.maximum_iterations)
    {
        check_cuda(cudaMemsetAsync(flags + 1, 0, sizeof(changed), storage.stream));
        check_cuda(static_cast<cudaError_t>(detail::cuda_sweep_site_esdf(
            device_keys, tsdf.size(), device_values, maximum_squared, flags + 1, storage.stream)));
        check_cuda(static_cast<cudaError_t>(detail::cuda_relax_site_esdf_neighbors(
            device_keys, tsdf.size(), device_values, maximum_squared, flags + 1, storage.stream)));
        check_cuda(cudaMemcpyAsync(&changed, flags + 1, sizeof(changed), cudaMemcpyDeviceToHost, storage.stream));
         check_cuda(cudaStreamSynchronize(storage.stream));
        statistics.downloaded_bytes += sizeof(changed);
        ++statistics.wave_iterations;
    }
    if (changed) throw std::runtime_error("CUDA site ESDF did not converge within its iteration limit");
    check_cuda(cudaMemcpyAsync(values.data(), device_values, bytes[2], cudaMemcpyDeviceToHost, storage.stream));
    check_cuda(cudaStreamSynchronize(storage.stream));
    statistics.downloaded_bytes += bytes[2];
    auto completed = detail::convert_cuda_site_esdf(keys, values, parameters);
    statistics.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    esdf.swap(completed);
    return statistics;
}
}
namespace covoxslam
{
cuda_blocks_statistics build_cuda_blocks_esdf(const tsdf_tiles& tsdf, esdf_tiles& esdf,
    const mapping_config& config, const cuda_blocks_options& options)
{
    const auto start = std::chrono::steady_clock::now();
    const auto difference = options.minimum_difference_m < 0 ? std::max(1.0e-5F, config.voxel_size * 1.0e-3F) : options.minimum_difference_m;
    const detail::cuda_blocks_parameters parameters{config.voxel_size, config.esdf_site_band_m,
        config.esdf_maximum_distance_m, config.esdf_minimum_weight, difference, options.bucket_count,
        options.distance == cuda_blocks_distance::full_euclidean};
    check_cuda(static_cast<cudaError_t>(detail::cuda_initialize_blocks(nullptr, nullptr, 0, parameters, {}, nullptr)));
    const auto bytes = detail::blocks_storage_bytes(tsdf.size(), options);
    if (!config.maximum_tiles || tsdf.size() > config.maximum_tiles) throw std::length_error("CUDA blocks tile limit exceeded");
    cuda_blocks_statistics statistics;
    for (const auto size : bytes) statistics.scratch_bytes += size;
     if (statistics.scratch_bytes > query_cuda_device(options.device).total_memory_bytes)
        throw std::length_error("CUDA blocks exceeds device memory");
    check_cuda(cudaSetDevice(options.device));
    if (tsdf.empty()) { esdf.clear(); return statistics; }
    std::vector<int3> order;
    for (const auto& [key, tile] : tsdf) order.push_back(key);
    std::sort(order.begin(), order.end(), block_index_less);
    std::vector<detail::cuda_tile_coordinate> keys;
    std::vector<detail::cuda_tsdf_tile> tiles(tsdf.size());
    std::vector<detail::cuda_blocks_voxel> values(tsdf.size() * 512);
    for (std::size_t slot = 0; slot < order.size(); ++slot)
    {
        const auto key = order[slot];
        keys.push_back({key.x, key.y, key.z});
        const auto& source = tsdf.at(key);
        for (std::size_t local = 0; local < 512; ++local) {
            const auto& voxel = source.voxels[local];
             tiles[slot].voxels[local] = {voxel.distance, voxel.weight, voxel.sample_count, voxel.flags};
        }
    }
    const auto canonical = detail::blocks_canonical_order(keys);
    detail::blocks_storage storage;
    check_cuda(cudaStreamCreateWithFlags(&storage.stream, cudaStreamNonBlocking));
    for (std::size_t index = 0; index < bytes.size(); ++index)
        if (bytes[index]) check_cuda(cudaMalloc(&storage.buffers[index], bytes[index]));
    const auto work = storage.workspace();
    auto* device_keys = storage.buffer<detail::cuda_tile_coordinate>(0);
    auto* device_tiles = storage.buffer<detail::cuda_tsdf_tile>(1);
    check_cuda(cudaMemcpyAsync(device_keys, keys.data(), bytes[0], cudaMemcpyHostToDevice, storage.stream));
    check_cuda(cudaMemcpyAsync(device_tiles, tiles.data(), bytes[1], cudaMemcpyHostToDevice, storage.stream));
    check_cuda(cudaMemcpyAsync(storage.buffers[5], canonical.data(), bytes[5], cudaMemcpyHostToDevice, storage.stream));
    statistics.uploaded_bytes = bytes[0] + bytes[1] + bytes[5];
    detail::cuda_blocks_queue state;
    const auto read_state = [&] {
        check_cuda(cudaMemcpyAsync(&state, work.queue, sizeof(state), cudaMemcpyDeviceToHost, storage.stream));
        check_cuda(cudaStreamSynchronize(storage.stream));
         statistics.downloaded_bytes += sizeof(state);
        if (state.status) throw std::runtime_error("CUDA blocks input or state invariant failed");
    };
    check_cuda(static_cast<cudaError_t>(detail::cuda_initialize_blocks(device_keys, device_tiles, tsdf.size(), parameters, work, storage.stream)));
    read_state();
    check_cuda(static_cast<cudaError_t>(detail::cuda_enqueue_blocks_seeds(values.size(), parameters, work, storage.stream)));
    read_state();
    while (state.size)
    {
        if (statistics.rounds >= options.maximum_rounds || state.pops >= options.maximum_queue_pops)
            throw std::runtime_error("CUDA blocks propagation limit exceeded");
        const auto remaining = options.maximum_queue_pops - state.pops;
        int error{};
        if (options.executor == cuda_blocks_executor::ordered)
            error = detail::cuda_propagate_blocks_ordered(device_keys, tsdf.size(), parameters, work,
                static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, options.pops_per_launch)), storage.stream);
        else
            error = detail::cuda_propagate_blocks_frontier(device_keys, tsdf.size(), parameters, work,
                static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, options.frontier_capacity)), storage.stream);
        check_cuda(static_cast<cudaError_t>(error));
         ++statistics.rounds;
        read_state();
    }
    check_cuda(static_cast<cudaError_t>(detail::cuda_validate_blocks(device_keys, device_tiles, tsdf.size(), parameters, work, storage.stream)));
    read_state();
    check_cuda(cudaMemcpyAsync(values.data(), work.values, bytes[2], cudaMemcpyDeviceToHost, storage.stream));
    check_cuda(cudaStreamSynchronize(storage.stream));
    statistics.downloaded_bytes += bytes[2];
    auto completed = detail::convert_cuda_blocks(keys, tiles, values, parameters);
    statistics.queue_pushes = state.pushes;
    statistics.queue_pops = state.pops;
    statistics.accepted_updates = state.accepted;
    statistics.candidate_relaxations = state.relaxations;
    statistics.eligible_voxels = state.eligible;
    statistics.fixed_seeds = state.seeds;
     statistics.elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    esdf.swap(completed);
    return statistics;
}
}
