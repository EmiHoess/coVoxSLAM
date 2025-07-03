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

cuda_device_info query_cuda_device(int index)
{
    int count{};
    check_cuda(cudaGetDeviceCount(&count));
     if (index < 0 || index >= count) throw std::out_of_range("CUDA device index is unavailable");
    cudaDeviceProp properties;
    check_cuda(cudaGetDeviceProperties(&properties, index));
    cuda_device_info result{index, properties.major, properties.minor};
    result.name = properties.name;
    result.total_memory_bytes = properties.totalGlobalMem;
    check_cuda(cudaRuntimeGetVersion(&result.runtime_version));
    check_cuda(cudaDriverGetVersion(&result.driver_version));
    return result;
}

operation_result<cuda_device_info> query_cuda_device_checked(int index)
{
    int count{};
    if (const auto status = cuda_status(cudaGetDeviceCount(&count)); !status) return std::unexpected(status.error());
    if (index < 0 || index >= count) return std::unexpected(operation_error{"CUDA device index is unavailable"});
    cudaDeviceProp properties{};
    if (const auto status = cuda_status(cudaGetDeviceProperties(&properties, index)); !status) return std::unexpected(status.error());
    cuda_device_info result{index, properties.major, properties.minor};
    result.name = properties.name;
    result.total_memory_bytes = properties.totalGlobalMem;
    if (const auto status = cuda_status(cudaRuntimeGetVersion(&result.runtime_version)); !status) return std::unexpected(status.error());
    if (const auto status = cuda_status(cudaDriverGetVersion(&result.driver_version)); !status) return std::unexpected(status.error());
    return result;
}

bool cuda_mapping_available(std::string* diagnostic)
{
    const auto result = query_cuda_device_checked(0);
    if (diagnostic) *diagnostic = result ? std::string{} : result.error().message;
    return result.has_value();
}

struct cuda_scan_context::implementation
{
    int device{};
    std::size_t capacity{};
    bool failed{};
    cudaStream_t stream{};
    std::array<void*, 3> buffers{};
     std::array<cudaEvent_t, 4> events{};
    ~implementation()
    {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        for (auto buffer : buffers) if (buffer) cudaFree(buffer);
        for (auto event : events) if (event) cudaEventDestroy(event);
        if (stream) cudaStreamDestroy(stream);
    }
};

cuda_scan_context::cuda_scan_context() = default;
operation_result<std::unique_ptr<cuda_scan_context>> cuda_scan_context::create(int device, std::size_t capacity)
{
    const auto info = query_cuda_device_checked(device);
    if (!info) return std::unexpected(info.error());
    const auto total = checked_cuda_storage_bytes(capacity);
    if (!total) return std::unexpected(total.error());
    if (*total > info->total_memory_bytes)
        return std::unexpected(operation_error{"CUDA scan storage exceeds total device memory"});
    auto owner = std::unique_ptr<cuda_scan_context>(new cuda_scan_context);
    auto& state = owner->state;
    state = std::make_unique<implementation>();
    state->device = device;
    state->capacity = capacity;
    if (const auto status = cuda_status(cudaSetDevice(device)); !status) return std::unexpected(status.error());
    if (const auto status = cuda_status(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking)); !status)
        return std::unexpected(status.error());
    const std::array<std::size_t, 3> bytes{capacity * 4, capacity * 8, (1 + (capacity - 1) / 256) * 8};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        if (const auto status = cuda_status(cudaMalloc(&state->buffers[index], bytes[index])); !status)
            return std::unexpected(status.error());
    for (auto& event : state->events) {
        if (const auto status = cuda_status(cudaEventCreate(&event)); !status) return std::unexpected(status.error());
    }
    return owner;
}

cuda_scan_context::~cuda_scan_context() = default;
std::size_t cuda_scan_context::capacity() const noexcept { return state->capacity; }

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


namespace covoxslam
{
void cuda_projective_mapper::integrate_scan(std::span<const ray> rays, const pose& scan_reference)
{
    if (!state->metadata->is_active() || state->failed) throw std::logic_error("projective mapping needs an active healthy submap");
    if (rays.size() > state->options.ray_capacity) throw std::length_error("projective scan exceeds its whole-scan ray capacity");
    const auto& rotation = scan_reference.rotation;
    const auto& translation = scan_reference.translation;
    const detail::cuda_projective_pose reference{{translation.x, translation.y, translation.z},
        {static_cast<float>(rotation.x), static_cast<float>(rotation.y), static_cast<float>(rotation.z), static_cast<float>(rotation.w)}};
    if (!detail::valid_projective_pose(reference)) throw std::invalid_argument("invalid projective scan reference pose");
    if (rays.empty()) return;
    const auto start = std::chrono::steady_clock::now();
    state->failed = true;
    check_cuda(cudaSetDevice(state->options.device));
    std::vector<detail::cuda_device_ray> input;
    input.reserve(rays.size());
    for (std::size_t index = 0; index < rays.size(); ++index) {
        const auto& ray = rays[index];
        if (ray.frame_id < 0 || ray.column > UINT32_MAX || ray.ring > UINT32_MAX)
            throw std::invalid_argument("projective ray metadata exceeds its native representation");
        input.push_back({{ray.origin.x, ray.origin.y, ray.origin.z}, {ray.endpoint.x, ray.endpoint.y, ray.endpoint.z},
            ray.weight, ray.stamp_ns, static_cast<std::uint32_t>(ray.frame_id), static_cast<std::uint32_t>(ray.column),
            static_cast<std::uint32_t>(ray.ring), static_cast<std::uint32_t>(index)});
    }
    const auto stream = state->stream;
    auto* device_rays = state->buffer<detail::cuda_device_ray>(0);
    auto* angles = state->buffer<float>(1);
    auto* winners = state->buffer<std::uint64_t>(2);
    auto* counts = state->buffer<std::uint32_t>(3);
    auto* offsets = state->buffer<std::uint64_t>(4);
    auto* summary = state->buffer<detail::cuda_ray_summary>(10);
    check_cuda(cudaMemcpyAsync(device_rays, input.data(), input.size() * sizeof(input[0]), cudaMemcpyHostToDevice, stream));
    state->statistics.cuda_uploaded_bytes += input.size() * sizeof(input[0]);
    check_cuda(cudaMemsetAsync(summary, 0, sizeof(*summary), stream));
    check_cuda(static_cast<cudaError_t>(detail::cuda_build_projective_range_image(device_rays, rays.size(), state->tracing,
        reference, state->calibration, angles, winners, state->pixels, &summary->status, stream)));
    state->checked_summary();
    const detail::cuda_projective_discovery_options discovery{state->calibration, angles,
        state->options.calibrated_policy, state->options.calibrated_policy ? state->options.raycast_subsampling_factor : 1U};
    check_cuda(static_cast<cudaError_t>(detail::cuda_count_projective_tiles(device_rays, rays.size(), winners, state->pixels,
        reference, state->tracing, counts, &summary->status, stream, discovery)));
    check_cuda(static_cast<cudaError_t>(detail::cuda_scan_counts(counts, offsets, state->buffer<std::uint64_t>(5), state->pixels, stream)));
    check_cuda(static_cast<cudaError_t>(detail::cuda_summarize_ray_contributions(counts, offsets, state->pixels, summary, stream)));
    const auto generated = state->checked_summary();
    if (generated.contributions > state->options.candidate_tile_capacity)
        throw std::length_error("projective tile candidates exceed their capacity");
    std::uint64_t updates{};
    if (generated.contributions) {
        check_cuda(static_cast<cudaError_t>(detail::cuda_emit_projective_tiles(device_rays, rays.size(), winners, state->pixels,
            reference, state->tracing, counts, offsets, state->buffer<detail::cuda_tile_coordinate>(6),
            state->options.candidate_tile_capacity, &summary->status, stream, discovery)));
        state->checked_summary();
        const auto selected = state->prepare_tiles(generated.contributions);
        const auto& config = state->config;
        check_cuda(static_cast<cudaError_t>(detail::cuda_fuse_projective_tiles(state->buffer<std::uint32_t>(8), selected,
            state->buffer<detail::cuda_tile_coordinate>(7), state->buffer<detail::cuda_tsdf_tile>(9),
            static_cast<std::uint32_t>(state->keys.size()), device_rays, rays.size(), winners, angles, reference,
            state->calibration, state->tracing, {config.truncation_distance, config.maximum_weight, config.extraction_distance},
            state->projection, state->buffer<std::uint64_t>(11), &summary->status, stream)));
        state->checked_summary();
        check_cuda(cudaMemcpyAsync(&updates, state->buffers[11], sizeof(updates), cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaStreamSynchronize(stream));
        state->statistics.cuda_downloaded_bytes += sizeof(updates);
    }
    auto& statistics = state->statistics;
    statistics.rays_submitted += rays.size();
    statistics.rays_integrated += generated.contributing_rays;
    statistics.rays_rejected += rays.size() - generated.contributing_rays;
    statistics.cuda_contributing_rays += generated.contributing_rays;
    statistics.voxel_updates += updates;
    statistics.allocated_tiles = state->keys.size();
    statistics.integration_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    state->failed = false;
}
}

namespace covoxslam
{
submap cuda_projective_mapper::finish(const submap_frame& frame)
{
    if (!state->metadata->is_active() || state->failed) throw std::logic_error("projective mapping needs an active healthy submap");
    auto validated_metadata = *state->metadata;
    auto result = validated_metadata.finish(frame);
    state->failed = true;
    check_cuda(cudaSetDevice(state->options.device));
    std::vector<detail::cuda_tsdf_tile> tiles(state->keys.size());
    const auto bytes = tiles.size() * sizeof(tiles[0]);
    if (bytes) {
        check_cuda(cudaMemcpyAsync(tiles.data(), state->buffers[9], bytes, cudaMemcpyDeviceToHost, state->stream));
        check_cuda(cudaStreamSynchronize(state->stream));
    }
    state->statistics.cuda_downloaded_bytes += bytes;
    result.config = state->config;
    result.statistics = state->statistics;
    for (std::size_t index = 0; index < tiles.size(); ++index) {
        tsdf_tile tile;
        bool observed{};
        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& source = tiles[index].voxels[local];
            if (!std::isfinite(source.distance) || !std::isfinite(source.weight) || source.weight < 0 ||
                source.weight > result.config.maximum_weight || std::abs(source.distance) > result.config.truncation_distance ||
                (source.flags & ~3U) || bool(source.flags & observed_flag) != (source.weight > 0) ||
                (!(source.flags & observed_flag) && (source.flags & near_surface_flag)) ||
                bool(source.flags & observed_flag) != (source.sample_count != 0))
                throw std::runtime_error("invalid canonical projective TSDF voxel");
            tile.voxels[local] = {source.distance, source.weight, source.sample_count, source.flags};
            observed = observed || (source.flags & observed_flag);
        }
        const auto key = state->keys[index];
        if (observed) result.tsdf.emplace(int3{key.x, key.y, key.z}, std::move(tile));
    }
    if (result.config.esdf_generator != esdf_backend::none) {
        const auto start = std::chrono::steady_clock::now();
        build_cpu_esdf(result.tsdf, result.esdf, result.config);
        result.statistics.esdf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
    state->metadata = std::make_unique<cpu_mapper>(std::move(validated_metadata));
    state->failed = false;
    return result;
}
}
