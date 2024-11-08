#include <covoxslam/cuda.hpp>
#include "sparse_mapping.h"
#include "cuda_site_conversion.hpp"
#include "cuda_voxblox_conversion.hpp"

#include <cuda_runtime_api.h>
#include "cuda_voxblox_storage.hpp"
#include <array>
#include <limits>
#include <stdexcept>

#include <type_traits>

namespace covoxslam
{
namespace
{
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
static_assert(std::is_standard_layout_v<detail::cuda_device_ray>);
static_assert(std::is_trivially_copyable_v<detail::cuda_device_ray>);
static_assert(sizeof(detail::cuda_device_ray) == 56);
static_assert(alignof(detail::cuda_device_ray) == 8);
static_assert(offsetof(detail::cuda_device_ray, stamp_ns) == 32);

std::size_t checked_cuda_storage_bytes(std::size_t capacity)
{
    if (!capacity || capacity > std::numeric_limits<std::uint32_t>::max())
        throw std::length_error("CUDA scan capacity is unsupported");
     const auto blocks = 1 + (capacity - 1) / 256;
    if (blocks > std::numeric_limits<std::size_t>::max() / 8 ||
        capacity > (std::numeric_limits<std::size_t>::max() - blocks * 8) / 12)
        throw std::length_error("CUDA scan byte count overflows addressable size");
    return capacity * 12 + blocks * 8;
}
}
}

namespace covoxslam
{
namespace
{
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

bool cuda_mapping_available(std::string* diagnostic) {
    try {
        query_cuda_device(0);
        if (diagnostic) diagnostic->clear();
        return true;
    }
    catch (const std::exception& error)
    {
         if (diagnostic) *diagnostic = error.what();
        return false;
    }
}

struct cuda_scan_context::implementation
{
    int device{};
    std::size_t capacity{};
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

cuda_scan_context::cuda_scan_context(int device, std::size_t capacity)
    : state(std::make_unique<implementation>())
{
    const auto info = query_cuda_device(device);
    if (checked_cuda_storage_bytes(capacity) > info.total_memory_bytes)
        throw std::length_error("CUDA scan storage exceeds total device memory");
    state->device = device;
    state->capacity = capacity;
    check_cuda(cudaSetDevice(device));
    check_cuda(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
    const std::array<std::size_t, 3> bytes{capacity * 4, capacity * 8, (1 + (capacity - 1) / 256) * 8};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        check_cuda(cudaMalloc(&state->buffers[index], bytes[index]));
    for (auto& event : state->events) check_cuda(cudaEventCreate(&event));
}

cuda_scan_context::~cuda_scan_context() = default;
std::size_t cuda_scan_context::capacity() const noexcept { return state->capacity; }

cuda_operation_times cuda_scan_context::exclusive_scan(
    std::span<const std::uint32_t> counts, std::span<std::uint64_t> offsets)
{
    if (counts.size() > state->capacity || offsets.size() < counts.size())
        throw std::length_error("CUDA scan buffers exceed their capacity");
     cuda_operation_times result;
    if (counts.empty()) return result;
    check_cuda(cudaSetDevice(state->device));
    check_cuda(cudaEventRecord(state->events[0], state->stream));
    check_cuda(cudaMemcpyAsync(state->buffers[0], counts.data(), counts.size_bytes(), cudaMemcpyHostToDevice, state->stream));
    check_cuda(cudaEventRecord(state->events[1], state->stream));
    check_cuda(static_cast<cudaError_t>(detail::cuda_scan_counts(
        static_cast<const std::uint32_t*>(state->buffers[0]), static_cast<std::uint64_t*>(state->buffers[1]),
        static_cast<std::uint64_t*>(state->buffers[2]), counts.size(), state->stream)));
    check_cuda(cudaEventRecord(state->events[2], state->stream));
    check_cuda(cudaMemcpyAsync(offsets.data(), state->buffers[1], counts.size() * 8, cudaMemcpyDeviceToHost, state->stream));
     check_cuda(cudaEventRecord(state->events[3], state->stream));
    check_cuda(cudaEventSynchronize(state->events[3]));
    check_cuda(cudaEventElapsedTime(&result.upload_ms, state->events[0], state->events[1]));
    check_cuda(cudaEventElapsedTime(&result.scan_ms, state->events[1], state->events[2]));
    check_cuda(cudaEventElapsedTime(&result.download_ms, state->events[2], state->events[3]));
    return result;
}
}

#include <vector>

namespace covoxslam
{
static_assert(sizeof(detail::cuda_ray_contribution) == 28);
static_assert(sizeof(detail::cuda_ray_summary) == 16);
struct cuda_ray_context::implementation
{
    int device{};
    std::size_t ray_capacity{};
    std::uint64_t contribution_capacity{};
    std::uint64_t contribution_count{};
     detail::cuda_ray_parameters parameters;
    cudaStream_t stream{};
    std::array<void*, 6> buffers{};
    std::array<cudaEvent_t, 6> events{};
    std::vector<detail::cuda_device_ray> rays_host;
    template<class value_type> value_type* buffer(std::size_t index) const noexcept
    {
        return static_cast<value_type*>(buffers[index]);
    }
    ~implementation()
    {
        if (!stream) return;
        cudaSetDevice(device);
        cudaStreamSynchronize(stream);
         for (auto buffer : buffers) if (buffer) cudaFree(buffer);
        for (auto event : events) if (event) cudaEventDestroy(event);
        cudaStreamDestroy(stream);
    }
};

cuda_ray_context::cuda_ray_context(int device, std::size_t ray_capacity,
    std::uint64_t contribution_capacity, cuda_ray_config config)
    : state(std::make_unique<implementation>())
{
    const auto info = query_cuda_device(device);
    state->parameters = {config.voxel_size, config.truncation_distance,
        config.minimum_ray_length_m, config.maximum_ray_length_m, config.voxel_carving, config.allow_clear};
    check_cuda(static_cast<cudaError_t>(detail::cuda_count_ray_contributions(
        nullptr, 0, state->parameters, nullptr, nullptr, nullptr)));
    if (!ray_capacity || ray_capacity > std::numeric_limits<std::uint32_t>::max() ||
        !contribution_capacity || contribution_capacity >
            std::numeric_limits<std::size_t>::max() / sizeof(detail::cuda_ray_contribution))
        throw std::length_error("CUDA contribution capacities are unsupported");
    const std::array<std::size_t, 6> bytes{ray_capacity * sizeof(detail::cuda_device_ray),
        ray_capacity * sizeof(std::uint32_t), ray_capacity * sizeof(std::uint64_t),
        (1 + (ray_capacity - 1) / 256) * sizeof(std::uint64_t),
        static_cast<std::size_t>(contribution_capacity) * sizeof(detail::cuda_ray_contribution),
        sizeof(detail::cuda_ray_summary)};
    std::size_t total{};
    for (const auto size : bytes) {
        if (size > std::numeric_limits<std::size_t>::max() - total)
            throw std::length_error("CUDA contribution storage overflows address space");
        total += size;
    }
    if (total > info.total_memory_bytes)
        throw std::length_error("CUDA contribution storage exceeds device memory");
     state->device = device;
    state->ray_capacity = ray_capacity;
    state->contribution_capacity = contribution_capacity;
    check_cuda(cudaSetDevice(device));
    check_cuda(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
    for (std::size_t index = 0; index < bytes.size(); ++index)
        check_cuda(cudaMalloc(&state->buffers[index], bytes[index]));
    for (auto& event : state->events) check_cuda(cudaEventCreate(&event));
}

cuda_ray_context::~cuda_ray_context() = default;

std::size_t cuda_ray_context::ray_capacity() const noexcept
{
    return state->ray_capacity;
}

std::uint64_t cuda_ray_context::contribution_capacity() const noexcept {
    return state->contribution_capacity;
}

detail::cuda_ray_device_view cuda_ray_context::device_view() const noexcept
{
    return {state->buffer<detail::cuda_ray_contribution>(4), state->contribution_count,
        state->contribution_capacity, state->buffer<std::uint32_t>(1), state->buffer<std::uint64_t>(2),
        state->buffer<std::uint64_t>(3), state->stream, state->device, state->ray_capacity};
}

void cuda_ray_context::copy_contributions(std::span<detail::cuda_ray_contribution> output)
{
    if (output.size() < state->contribution_count)
        throw std::length_error("CUDA contribution download buffer is too small");
    if (!state->contribution_count) return;
     check_cuda(cudaSetDevice(state->device));
    check_cuda(cudaMemcpyAsync(output.data(), state->buffers[4],
        static_cast<std::size_t>(state->contribution_count) * sizeof(detail::cuda_ray_contribution),
        cudaMemcpyDeviceToHost, state->stream));
    check_cuda(cudaStreamSynchronize(state->stream));
}
}

#include <covoxslam/mapping.hpp>

namespace covoxslam
{
cuda_contribution_statistics cuda_ray_context::generate(std::span<const ray> rays)
{
    state->contribution_count = 0;
    if (rays.size() > state->ray_capacity)
        throw std::length_error("CUDA ray batch exceeds its capacity");
    check_cuda(cudaSetDevice(state->device));
    check_cuda(cudaStreamSynchronize(state->stream));
    state->rays_host.resize(rays.size());
    for (std::size_t index = 0; index < rays.size(); ++index)
    {
        const auto& input = rays[index];
        if (input.frame_id < 0 || input.column > std::numeric_limits<std::uint32_t>::max() ||
            input.ring > std::numeric_limits<std::uint32_t>::max())
            throw std::out_of_range("CUDA ray provenance is not representable");
         state->rays_host[index] = {{input.origin.x, input.origin.y, input.origin.z},
            {input.endpoint.x, input.endpoint.y, input.endpoint.z}, input.weight, input.stamp_ns,
            static_cast<std::uint32_t>(input.frame_id), static_cast<std::uint32_t>(input.column),
            static_cast<std::uint32_t>(input.ring), static_cast<std::uint32_t>(index)};
    }
    cuda_contribution_statistics result;
    result.submitted_rays = rays.size();
    if (rays.empty()) return result;
    auto* device_rays = state->buffer<detail::cuda_device_ray>(0);
    auto* counts = state->buffer<std::uint32_t>(1);
    auto* offsets = state->buffer<std::uint64_t>(2);
    auto* sums = state->buffer<std::uint64_t>(3);
    auto* contributions = state->buffer<detail::cuda_ray_contribution>(4);
     auto* summary = state->buffer<detail::cuda_ray_summary>(5);
    result.uploaded_bytes = rays.size() * sizeof(detail::cuda_device_ray);
    check_cuda(cudaEventRecord(state->events[0], state->stream));
    check_cuda(cudaMemcpyAsync(device_rays, state->rays_host.data(),
        static_cast<std::size_t>(result.uploaded_bytes), cudaMemcpyHostToDevice, state->stream));
    check_cuda(cudaMemsetAsync(summary, 0, sizeof(*summary), state->stream));
    check_cuda(cudaEventRecord(state->events[1], state->stream));
    check_cuda(static_cast<cudaError_t>(detail::cuda_count_ray_contributions(
        device_rays, rays.size(), state->parameters, counts, &summary->status, state->stream)));
    check_cuda(cudaEventRecord(state->events[2], state->stream));
    check_cuda(static_cast<cudaError_t>(detail::cuda_scan_counts(counts, offsets, sums, rays.size(), state->stream)));
    check_cuda(static_cast<cudaError_t>(detail::cuda_summarize_ray_contributions(
        counts, offsets, rays.size(), summary, state->stream)));
    check_cuda(cudaEventRecord(state->events[3], state->stream));
    detail::cuda_ray_summary completed;
    check_cuda(cudaMemcpyAsync(&completed, summary, sizeof(completed), cudaMemcpyDeviceToHost, state->stream));
     check_cuda(cudaStreamSynchronize(state->stream));
    if (completed.status) throw std::runtime_error("CUDA contribution counting failed");
    if (completed.contributions > state->contribution_capacity)
        throw std::length_error("CUDA contribution batch exceeds its capacity");
    check_cuda(cudaEventRecord(state->events[4], state->stream));
    check_cuda(static_cast<cudaError_t>(detail::cuda_emit_ray_contributions(device_rays, rays.size(),
        state->parameters, counts, offsets, contributions, state->contribution_capacity, &summary->status, state->stream)));
    check_cuda(cudaEventRecord(state->events[5], state->stream));
    check_cuda(cudaMemcpyAsync(&completed.status, &summary->status,
        sizeof(completed.status), cudaMemcpyDeviceToHost, state->stream));
    check_cuda(cudaStreamSynchronize(state->stream));
    if (completed.status) throw std::runtime_error("CUDA contribution emission failed");
    check_cuda(cudaEventElapsedTime(&result.upload_ms, state->events[0], state->events[1]));
    check_cuda(cudaEventElapsedTime(&result.count_ms, state->events[1], state->events[2]));
    check_cuda(cudaEventElapsedTime(&result.scan_ms, state->events[2], state->events[3]));
     check_cuda(cudaEventElapsedTime(&result.emit_ms, state->events[4], state->events[5]));
    result.contributing_rays = completed.contributing_rays;
    result.contributions = completed.contributions;
    result.downloaded_bytes = sizeof(completed) + sizeof(completed.status);
    state->contribution_count = completed.contributions;
    return result;
}
}

#include <algorithm>
#include <chrono>
#include <cmath>

namespace covoxslam
{
struct cuda_mapper::implementation
{
    mapping_config config;
    cuda_mapper_options options;
    std::unique_ptr<cuda_ray_context> rays;
    std::array<void*, 8> buffers{};
     std::optional<submap_frame> active;
    mapping_statistics statistics;
    std::vector<detail::cuda_tile_coordinate> keys;
    std::unordered_map<int3, std::uint32_t, int3_hash> slots;
    std::vector<detail::cuda_voxel_segment> segments;
    bool failed{};
    template<class value_type> value_type* buffer(std::size_t index) const noexcept
    {
        return static_cast<value_type*>(buffers[index]);
    }
    std::uint32_t prepare_segments(detail::cuda_ray_device_view view);
    ~implementation()
    {
        if (!rays) return;
        cudaSetDevice(options.device);
        cudaStreamSynchronize(static_cast<cudaStream_t>(rays->device_view().stream));
         for (auto buffer : buffers) if (buffer) cudaFree(buffer);
    }
};

cuda_mapper::cuda_mapper(mapping_config config, cuda_mapper_options options)
    : state(std::make_unique<implementation>())
{
    if (!config.valid()) throw std::invalid_argument("invalid CUDA mapping configuration");
    if (config.esdf_generator != esdf_backend::none && config.esdf_generator != esdf_backend::cpu_voxblox_batch)
        throw std::invalid_argument("CUDA ESDF requires the external submap finalizer");
    const auto limit = std::numeric_limits<std::uint32_t>::max();
    if (!options.ray_capacity || options.ray_capacity > limit || !options.contribution_capacity ||
        options.contribution_capacity > limit || config.maximum_tiles > limit)
        throw std::length_error("CUDA mapper capacities are unsupported");
    const auto count = options.contribution_capacity;
    const std::array<std::size_t, 8> bytes{count * sizeof(detail::cuda_ray_contribution), count * 4,
        count * 8, (1 + (count - 1) / 256) * 8, count * sizeof(detail::cuda_voxel_segment),
        config.maximum_tiles * sizeof(detail::cuda_tile_coordinate),
        config.maximum_tiles * sizeof(detail::cuda_tsdf_tile), 8};
    std::uint64_t total = options.ray_capacity * (sizeof(detail::cuda_device_ray) + 12) +
        (1 + (options.ray_capacity - 1) / 256) * 8 + count * sizeof(detail::cuda_ray_contribution) +
        sizeof(detail::cuda_ray_summary);
    for (const auto size : bytes) total += size;
     if (total > options.maximum_device_bytes)
        throw std::length_error("CUDA mapper exceeds its device storage budget");
    const auto device = query_cuda_device(options.device);
    if (total > device.total_memory_bytes) throw std::length_error("CUDA mapper exceeds device memory");
    state->config = config;
    state->options = options;
    state->rays = std::make_unique<cuda_ray_context>(options.device, options.ray_capacity, count,
        cuda_ray_config{config.voxel_size, config.truncation_distance,
                        config.minimum_ray_length_m, config.maximum_ray_length_m});
    check_cuda(cudaSetDevice(options.device));
    for (std::size_t index = 0; index < bytes.size(); ++index)
        check_cuda(cudaMalloc(&state->buffers[index], bytes[index]));
}

cuda_mapper::~cuda_mapper() = default;

void cuda_mapper::begin(const submap_frame& frame)
{
    if (state->active) throw std::logic_error("finish the active submap before beginning another");
    if (frame.end_stamp_ns < frame.start_stamp_ns ||
        !std::isfinite(frame.anchor.x) || !std::isfinite(frame.anchor.y) ||
        !std::isfinite(frame.anchor.z) || !std::isfinite(frame.anchor.yaw))
        throw std::invalid_argument("invalid submap anchor or interval");
    state->active = frame;
    state->keys.clear();
     state->slots.clear();
    state->segments.clear();
    state->statistics = {};
    state->failed = false;
}

bool cuda_mapper::is_active() const noexcept { return state->active.has_value(); }
}

namespace covoxslam
{
std::uint32_t cuda_mapper::implementation::prepare_segments(detail::cuda_ray_device_view view)
{
    const auto stream = static_cast<cudaStream_t>(view.stream);
    auto* flags = buffer<std::uint32_t>(1);
    auto* offsets = buffer<std::uint64_t>(2);
    auto* sums = buffer<std::uint64_t>(3);
    auto* device_segments = buffer<detail::cuda_voxel_segment>(4);
     auto* counters = buffer<std::uint32_t>(7);
    check_cuda(cudaMemsetAsync(counters, 0, 8, stream));
    check_cuda(static_cast<cudaError_t>(detail::cuda_stable_sort_contributions(view.contributions,
        buffer<detail::cuda_ray_contribution>(0), flags, offsets, sums, view.count, stream)));
    check_cuda(static_cast<cudaError_t>(detail::cuda_build_voxel_segments(view.contributions, view.count,
        flags, offsets, sums, device_segments, static_cast<std::uint32_t>(options.contribution_capacity),
        counters, counters + 1, stream)));
    std::array<std::uint32_t, 2> result{};
    check_cuda(cudaMemcpyAsync(result.data(), counters, sizeof(result), cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaStreamSynchronize(stream));
    statistics.cuda_downloaded_bytes += sizeof(result);
    if (result[1] || !result[0] || result[0] > view.count) throw std::runtime_error("CUDA contribution grouping failed");
    segments.resize(result[0]);
    const auto segment_bytes = segments.size() * sizeof(segments[0]);
     check_cuda(cudaMemcpyAsync(segments.data(), device_segments, segment_bytes, cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaStreamSynchronize(stream));
    statistics.cuda_downloaded_bytes += segment_bytes;
    auto proposed_keys = keys;
    auto proposed_slots = slots;
    for (auto& segment : segments)
    {
        const int3 voxel{segment.voxel[0], segment.voxel[1], segment.voxel[2]};
        const auto key = block_index(voxel);
        auto found = proposed_slots.find(key);
        if (found == proposed_slots.end()) {
            if (proposed_keys.size() >= config.maximum_tiles)
                throw std::length_error("CUDA map exceeds its tile capacity");
            const auto slot = static_cast<std::uint32_t>(proposed_keys.size());
            proposed_keys.push_back({key.x, key.y, key.z});
             found = proposed_slots.emplace(key, slot).first;
        }
        segment.tile_slot = found->second;
        segment.local_index = static_cast<std::uint32_t>(local_index(voxel));
    }
    const auto previous = keys.size();
    keys = std::move(proposed_keys);
    slots = std::move(proposed_slots);
    const auto added = keys.size() - previous;
    if (added)
    {
        check_cuda(cudaMemcpyAsync(buffer<detail::cuda_tile_coordinate>(5) + previous, keys.data() + previous,
            added * sizeof(keys[0]), cudaMemcpyHostToDevice, stream));
        check_cuda(cudaMemsetAsync(buffer<detail::cuda_tsdf_tile>(6) + previous, 0,
            added * sizeof(detail::cuda_tsdf_tile), stream));
        statistics.cuda_uploaded_bytes += added * sizeof(keys[0]);
    }
    check_cuda(cudaMemcpyAsync(device_segments, segments.data(), segment_bytes, cudaMemcpyHostToDevice, stream));
     statistics.cuda_uploaded_bytes += segment_bytes;
    return result[0];
}
}

namespace covoxslam
{
void cuda_mapper::integrate_scan(std::span<const ray> rays)
{
    if (!state->active || state->failed) throw std::logic_error("CUDA mapping needs an active healthy submap");
    const auto start = std::chrono::steady_clock::now();
    state->failed = true;
    for (std::size_t first = 0; first < rays.size();)
    {
        const auto batch = rays.subspan(first, std::min(state->options.ray_capacity, rays.size() - first));
        const auto generated = state->rays->generate(batch);
        const auto view = state->rays->device_view();
        const auto stream = static_cast<cudaStream_t>(view.stream);
        auto& statistics = state->statistics;
        statistics.cuda_uploaded_bytes += generated.uploaded_bytes;
         statistics.cuda_downloaded_bytes += generated.downloaded_bytes;
        if (view.count)
        {
            const auto segments = state->prepare_segments(view);
            auto* status = state->buffer<std::uint32_t>(7) + 1;
            const auto& config = state->config;
            check_cuda(static_cast<cudaError_t>(detail::cuda_fuse_voxel_segments(view.contributions,
                view.count, state->buffer<detail::cuda_voxel_segment>(4), segments,
                state->buffer<detail::cuda_tile_coordinate>(5), state->buffer<detail::cuda_tsdf_tile>(6),
                static_cast<std::uint32_t>(state->keys.size()),
                {config.truncation_distance, config.maximum_weight, config.extraction_distance}, status, stream)));
            std::uint32_t error{};
            check_cuda(cudaMemcpyAsync(&error, status, sizeof(error), cudaMemcpyDeviceToHost, stream));
            check_cuda(cudaStreamSynchronize(stream));
            statistics.cuda_downloaded_bytes += sizeof(error);
            if (error) throw std::runtime_error("CUDA TSDF fusion failed");
             statistics.cuda_voxel_segments += segments;
        }
        std::uint64_t accepted{};
        for (const auto& input : batch) {
            const float distance = length(input.endpoint - input.origin);
            if (std::isfinite(distance) && distance >= state->config.minimum_ray_length_m &&
                distance <= state->config.maximum_ray_length_m && std::isfinite(input.weight) && input.weight > 0 &&
                std::isfinite(input.origin.x) && std::isfinite(input.origin.y) && std::isfinite(input.origin.z)) ++accepted;
        }
        statistics.rays_submitted += batch.size();
        statistics.rays_integrated += accepted;
        statistics.rays_rejected += batch.size() - accepted;
        statistics.voxel_updates += generated.contributions;
        statistics.cuda_contributing_rays += generated.contributing_rays;
        statistics.allocated_tiles = state->keys.size();
        first += batch.size();
    }
    state->statistics.integration_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    state->failed = false;
}

submap cuda_mapper::finish(const submap_frame& frame)
{
    if (!state->active || state->failed) throw std::logic_error("CUDA mapping needs an active healthy submap");
    const auto& previous = *state->active;
    if (frame.id != previous.id || frame.start_stamp_ns != previous.start_stamp_ns ||
        frame.end_stamp_ns < previous.end_stamp_ns || frame.end_stamp_ns < frame.start_stamp_ns ||
        frame.anchor.x != previous.anchor.x || frame.anchor.y != previous.anchor.y ||
        frame.anchor.z != previous.anchor.z || frame.anchor.yaw != previous.anchor.yaw ||
        frame.frame_ids.size() < previous.frame_ids.size() ||
        !std::equal(previous.frame_ids.begin(), previous.frame_ids.end(), frame.frame_ids.begin()) ||
        std::any_of(frame.frame_ids.begin(), frame.frame_ids.end(), [](int id) { return id < 0; }) ||
        std::adjacent_find(frame.frame_ids.begin(), frame.frame_ids.end(), std::greater_equal<int>{}) != frame.frame_ids.end())
        throw std::logic_error("CUDA submap finalization does not match its anchor or frame history");
    state->failed = true;
    submap result;
    result.frame = frame;
    result.optimized_world_pose = frame.anchor;
    result.config = state->config;
    std::vector<detail::cuda_tsdf_tile> tiles(state->keys.size());
    check_cuda(cudaSetDevice(state->options.device));
    const auto stream = static_cast<cudaStream_t>(state->rays->device_view().stream);
    if (!tiles.empty()) {
        check_cuda(cudaMemcpyAsync(tiles.data(), state->buffers[6], tiles.size() * sizeof(tiles[0]),
            cudaMemcpyDeviceToHost, stream));
        check_cuda(cudaStreamSynchronize(stream));
    }
    state->statistics.cuda_downloaded_bytes += tiles.size() * sizeof(detail::cuda_tsdf_tile);
    for (std::size_t index = 0; index < tiles.size(); ++index)
    {
        tsdf_tile tile;
        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& source = tiles[index].voxels[local];
            if (!std::isfinite(source.distance) || !std::isfinite(source.weight) || source.weight < 0 ||
                source.weight > state->config.maximum_weight || (source.flags & ~3U) ||
                std::abs(source.distance) > state->config.truncation_distance)
                throw std::runtime_error("invalid canonical CUDA TSDF voxel");
             tile.voxels[local] = {source.distance, source.weight, source.sample_count, source.flags};
        }
        const auto& key = state->keys[index];
        result.tsdf.emplace(int3{key.x, key.y, key.z}, std::move(tile));
    }
    result.statistics = state->statistics;
    if (state->config.esdf_generator == esdf_backend::cpu_voxblox_batch) {
        const auto start = std::chrono::steady_clock::now();
        build_cpu_esdf(result.tsdf, result.esdf, result.config);
        result.statistics.esdf_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
    state->active.reset();
    state->failed = false;
    return result;
}
}

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
cuda_voxblox_statistics build_cuda_voxblox_esdf(const tsdf_tiles& tsdf, esdf_tiles& esdf,
    const mapping_config& config, const cuda_voxblox_options& options)
{
    const auto start = std::chrono::steady_clock::now();
    const auto difference = options.minimum_difference_m < 0 ? std::max(1.0e-5F, config.voxel_size * 1.0e-3F) : options.minimum_difference_m;
    const detail::cuda_voxblox_parameters parameters{config.voxel_size, config.esdf_site_band_m,
        config.esdf_maximum_distance_m, config.esdf_minimum_weight, difference, options.bucket_count,
        options.distance == cuda_voxblox_distance::full_euclidean};
    check_cuda(static_cast<cudaError_t>(detail::cuda_initialize_voxblox(nullptr, nullptr, 0, parameters, {}, nullptr)));
    const auto bytes = detail::voxblox_storage_bytes(tsdf.size(), options);
    if (!config.maximum_tiles || tsdf.size() > config.maximum_tiles) throw std::length_error("CUDA Voxblox tile limit exceeded");
    cuda_voxblox_statistics statistics;
    for (const auto size : bytes) statistics.scratch_bytes += size;
     if (statistics.scratch_bytes > query_cuda_device(options.device).total_memory_bytes)
        throw std::length_error("CUDA Voxblox exceeds device memory");
    check_cuda(cudaSetDevice(options.device));
    if (tsdf.empty()) { esdf.clear(); return statistics; }
    std::vector<int3> order;
    for (const auto& [key, tile] : tsdf) order.push_back(key);
    std::sort(order.begin(), order.end(), block_index_less);
    std::vector<detail::cuda_tile_coordinate> keys;
    std::vector<detail::cuda_tsdf_tile> tiles(tsdf.size());
    std::vector<detail::cuda_voxblox_voxel> values(tsdf.size() * 512);
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
    const auto canonical = detail::voxblox_canonical_order(keys);
    detail::voxblox_storage storage;
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
    detail::cuda_voxblox_queue state;
    const auto read_state = [&] {
        check_cuda(cudaMemcpyAsync(&state, work.queue, sizeof(state), cudaMemcpyDeviceToHost, storage.stream));
        check_cuda(cudaStreamSynchronize(storage.stream));
         statistics.downloaded_bytes += sizeof(state);
        if (state.status) throw std::runtime_error("CUDA Voxblox input or state invariant failed");
    };
    check_cuda(static_cast<cudaError_t>(detail::cuda_initialize_voxblox(device_keys, device_tiles, tsdf.size(), parameters, work, storage.stream)));
    read_state();
    check_cuda(static_cast<cudaError_t>(detail::cuda_enqueue_voxblox_seeds(values.size(), parameters, work, storage.stream)));
    read_state();
    while (state.size)
    {
        if (statistics.rounds >= options.maximum_rounds || state.pops >= options.maximum_queue_pops)
            throw std::runtime_error("CUDA Voxblox propagation limit exceeded");
        const auto remaining = options.maximum_queue_pops - state.pops;
        int error{};
        if (options.executor == cuda_voxblox_executor::ordered)
            error = detail::cuda_propagate_voxblox_ordered(device_keys, tsdf.size(), parameters, work,
                static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, options.pops_per_launch)), storage.stream);
        else
            error = detail::cuda_propagate_voxblox_frontier(device_keys, tsdf.size(), parameters, work,
                static_cast<std::uint32_t>(std::min<std::uint64_t>(remaining, options.frontier_capacity)), storage.stream);
        check_cuda(static_cast<cudaError_t>(error));
         ++statistics.rounds;
        read_state();
    }
    check_cuda(static_cast<cudaError_t>(detail::cuda_validate_voxblox(device_keys, device_tiles, tsdf.size(), parameters, work, storage.stream)));
    read_state();
    check_cuda(cudaMemcpyAsync(values.data(), work.values, bytes[2], cudaMemcpyDeviceToHost, storage.stream));
    check_cuda(cudaStreamSynchronize(storage.stream));
    statistics.downloaded_bytes += bytes[2];
    auto completed = detail::convert_cuda_voxblox(keys, tiles, values, parameters);
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

#include "cuda_projective_internal.hpp"

namespace covoxslam
{
cuda_projective_mapper::cuda_projective_mapper(const sensor& sensor, mapping_config config, cuda_projective_options options)
    : state(std::make_unique<implementation>())
{
    static_cast<void>(cpu_mapper(config));
    const auto finite_nonnegative = [](float value) { return std::isfinite(value) && value >= 0; };
    if ((options.interpolation != cuda_projective_interpolation::nearest &&
         options.interpolation != cuda_projective_interpolation::linear_nearest) ||
        !finite_nonnegative(options.depth_discontinuity_voxels) || !finite_nonnegative(options.nearest_ray_distance_voxels) ||
        !std::isfinite(options.depth_discontinuity_voxels * config.voxel_size) ||
        !std::isfinite(options.nearest_ray_distance_voxels * config.voxel_size) ||
        !std::isfinite(options.sparsity_compensation_factor) || options.sparsity_compensation_factor <= 0 ||
        (options.weight_dropoff && config.truncation_distance <= config.voxel_size) ||
        (double(config.maximum_ray_length_m) + config.truncation_distance) / config.voxel_size >= 10000000)
        throw std::invalid_argument("invalid projective interpolation or weighting controls");
    if (sensor.rings > INT32_MAX || sensor.columns > INT32_MAX)
        throw std::length_error("projective calibration exceeds index capacity");
    state->calibration = {static_cast<std::uint32_t>(sensor.rings), static_cast<std::uint32_t>(sensor.columns),
        static_cast<float>(sensor.azimuth_start_rad), static_cast<float>(sensor.azimuth_coverage_rad), sensor.counter_clockwise};
    if (!detail::valid_projective_calibration(state->calibration, sensor.vertical_angles_rad.data(), sensor.vertical_angles_rad.size()))
        throw std::invalid_argument("invalid projective sensor calibration");
    state->pixels = sensor.rings * sensor.columns;
    if (!options.ray_capacity) options.ray_capacity = state->pixels;
    if (state->pixels > UINT32_MAX || options.ray_capacity > UINT32_MAX || !options.candidate_tile_capacity ||
        options.candidate_tile_capacity > UINT32_MAX || config.maximum_tiles > INT32_MAX)
        throw std::length_error("projective capacities exceed native limits");
    state->projection = {static_cast<detail::cuda_projective_interpolation>(options.interpolation),
        options.depth_discontinuity_voxels, options.nearest_ray_distance_voxels, options.constant_weight,
        options.weight_dropoff, options.sparsity_compensation, options.sparsity_compensation_factor};
    state->tracing = {config.voxel_size, config.truncation_distance, config.minimum_ray_length_m,
        config.maximum_ray_length_m, options.voxel_carving, options.allow_clear};
    const auto pixels = state->pixels;
    const std::array<std::size_t, 12> bytes{options.ray_capacity * sizeof(detail::cuda_device_ray), sensor.rings * 4,
        pixels * 8, pixels * 4, pixels * 8, (1 + (pixels - 1) / 256) * 8,
        options.candidate_tile_capacity * sizeof(detail::cuda_tile_coordinate), config.maximum_tiles * sizeof(detail::cuda_tile_coordinate),
        config.maximum_tiles * 4, config.maximum_tiles * sizeof(detail::cuda_tsdf_tile), sizeof(detail::cuda_ray_summary), 8};
    std::size_t total{};
    for (const auto size : bytes) {
        if (size > options.maximum_device_bytes - total) throw std::length_error("projective device budget exceeded");
         total += size;
    }
    if (total > query_cuda_device(options.device).total_memory_bytes) throw std::length_error("projective device memory exceeded");
    state->config = config;
    state->options = options;
    auto metadata_config = config;
    metadata_config.esdf_generator = esdf_backend::none;
    state->metadata = std::make_unique<cpu_mapper>(metadata_config);
    check_cuda(cudaSetDevice(options.device));
    check_cuda(cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking));
    for (std::size_t index = 0; index < bytes.size(); ++index) check_cuda(cudaMalloc(&state->buffers[index], bytes[index]));
    check_cuda(cudaMemcpyAsync(state->buffers[1], sensor.vertical_angles_rad.data(), bytes[1], cudaMemcpyHostToDevice, state->stream));
    check_cuda(cudaStreamSynchronize(state->stream));
    state->setup_uploaded_bytes = bytes[1];
}

cuda_projective_mapper::~cuda_projective_mapper() = default;
}

namespace covoxslam
{
void cuda_projective_mapper::begin(const submap_frame& frame)
{
    state->metadata->begin(frame);
    state->keys.clear();
    state->slots.clear();
    state->statistics = {};
    state->statistics.cuda_uploaded_bytes = state->setup_uploaded_bytes;
    state->setup_uploaded_bytes = 0;
    state->failed = false;
}

bool cuda_projective_mapper::is_active() const noexcept { return state->metadata->is_active(); }

detail::cuda_ray_summary cuda_projective_mapper::implementation::checked_summary()
{
    detail::cuda_ray_summary completed;
    check_cuda(cudaMemcpyAsync(&completed, buffers[10], sizeof(completed), cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaStreamSynchronize(stream));
    statistics.cuda_downloaded_bytes += sizeof(completed);
    if (completed.status) throw std::runtime_error("projective device operation failed");
    return completed;
}

std::size_t cuda_projective_mapper::implementation::prepare_tiles(std::uint64_t count)
{
    if (!count) return 0;
    std::vector<detail::cuda_tile_coordinate> candidates(static_cast<std::size_t>(count));
    const auto bytes = candidates.size() * sizeof(candidates[0]);
    check_cuda(cudaMemcpyAsync(candidates.data(), buffers[6], bytes, cudaMemcpyDeviceToHost, stream));
    check_cuda(cudaStreamSynchronize(stream));
    statistics.cuda_downloaded_bytes += bytes;
    const auto less = [](const auto& left, const auto& right) {
        return block_index_less({left.x, left.y, left.z}, {right.x, right.y, right.z});
    };
    std::sort(candidates.begin(), candidates.end(), less);
    candidates.erase(std::unique(candidates.begin(), candidates.end(), [&](const auto& left, const auto& right) {
        return !less(left, right) && !less(right, left);
    }), candidates.end());
    auto proposed_keys = keys;
    auto proposed_slots = slots;
    std::vector<std::uint32_t> selected;
    for (const auto& candidate : candidates) {
        const int3 key{candidate.x, candidate.y, candidate.z};
        auto found = proposed_slots.find(key);
        if (found == proposed_slots.end()) {
            if (proposed_keys.size() >= config.maximum_tiles) throw std::length_error("projective map tile limit exceeded");
            const auto slot = static_cast<std::uint32_t>(proposed_keys.size());
            proposed_keys.push_back(candidate);
            found = proposed_slots.emplace(key, slot).first;
        }
        selected.push_back(found->second);
    }
    std::sort(selected.begin(), selected.end());
    const auto previous = keys.size();
    keys = std::move(proposed_keys);
    slots = std::move(proposed_slots);
    const auto added = keys.size() - previous;
    if (added) {
        check_cuda(cudaMemcpyAsync(buffer<detail::cuda_tile_coordinate>(7) + previous, keys.data() + previous,
            added * sizeof(keys[0]), cudaMemcpyHostToDevice, stream));
        check_cuda(cudaMemsetAsync(buffer<detail::cuda_tsdf_tile>(9) + previous, 0, added * sizeof(detail::cuda_tsdf_tile), stream));
        statistics.cuda_uploaded_bytes += added * sizeof(keys[0]);
    }
    check_cuda(cudaMemcpyAsync(buffers[8], selected.data(), selected.size() * sizeof(selected[0]), cudaMemcpyHostToDevice, stream));
    check_cuda(cudaStreamSynchronize(stream));
    statistics.cuda_uploaded_bytes += selected.size() * sizeof(selected[0]);
    return selected.size();
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
    check_cuda(static_cast<cudaError_t>(detail::cuda_count_projective_tiles(device_rays, rays.size(), winners, state->pixels,
        reference, state->tracing, counts, &summary->status, stream)));
    check_cuda(static_cast<cudaError_t>(detail::cuda_scan_counts(counts, offsets, state->buffer<std::uint64_t>(5), state->pixels, stream)));
    check_cuda(static_cast<cudaError_t>(detail::cuda_summarize_ray_contributions(counts, offsets, state->pixels, summary, stream)));
    const auto generated = state->checked_summary();
    if (generated.contributions > state->options.candidate_tile_capacity)
        throw std::length_error("projective tile candidates exceed their capacity");
    std::uint64_t updates{};
    if (generated.contributions) {
        check_cuda(static_cast<cudaError_t>(detail::cuda_emit_projective_tiles(device_rays, rays.size(), winners, state->pixels,
            reference, state->tracing, counts, offsets, state->buffer<detail::cuda_tile_coordinate>(6),
            state->options.candidate_tile_capacity, &summary->status, stream)));
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
