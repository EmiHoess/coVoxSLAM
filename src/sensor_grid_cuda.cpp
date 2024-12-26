#include <covoxslam/sensor_grid_cuda.hpp>
#include <covoxslam/sensor_grid_precision.h>
#include <covoxslam/sensor_grid_layout.hpp>
#include "sensor_grid/native.h"
#include <cuda_runtime_api.h>
#include <array>
#include <algorithm>

namespace covoxslam::sensor_grid
{
struct gpu_grid::implementation
{
    int device{};
    geometry grid{};
    lookup_view table{};
    layout_view layout{};
    gather_options options{};
    prepared_ray* prepared{};
    float* ranges{};
     voxel* output{};
    cudaStream_t stream{};
    std::array<cudaEvent_t, 3> events{};
    std::array<cudaEvent_t, 5> bin_events{};
    std::array<void*, 13> allocations{};
    std::size_t count{}, bytes{};
    bool selected{}, failed{}, ready{};
    operation_status check(int code)
    {
        if (code) {
            failed = true;
            ready = false;
            return std::unexpected(operation_error{cudaGetErrorString(static_cast<cudaError_t>(code))});
        }
        return {};
    }
    template<class value_type>
    operation_status allocate(value_type*& pointer, std::size_t size)
    {
        const auto status = check(cudaMalloc(reinterpret_cast<void**>(&pointer), size * sizeof(value_type)));

        if (pointer) allocations[count++] = pointer;
        return status;
    }
    template<class value_type>
    operation_status upload(const std::vector<value_type>& source, const value_type*& target)
    {
        value_type* pointer{};
        auto status = allocate(pointer, source.size());
        target = pointer;
        if (status) status = check(cudaMemcpyAsync(pointer, source.data(), source.size() * sizeof(value_type), cudaMemcpyHostToDevice, stream));
        return status;
    }
    ~implementation()
    {
        if (!selected) return;
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);

        for (auto event : events)
            if (event) cudaEventDestroy(event);

        for (auto event : bin_events)
            if (event) cudaEventDestroy(event);

        for (std::size_t i = 0; i < count; ++i) cudaFree(allocations[i]);
        if (stream) cudaStreamDestroy(stream);
    }
};
gpu_grid::gpu_grid() = default;
gpu_grid::~gpu_grid() = default;
std::size_t gpu_grid::device_bytes() const noexcept { return state->bytes; }
int gpu_grid::device_index() const noexcept { return state->device; }
field_view gpu_grid::device_field() const noexcept
{
    if (!state->ready || state->failed) return {};
    return {state->output, state->grid.side, state->table.settings.voxel_size, state->grid.origin};
}
operation_result<update_times> gpu_grid::update(std::span<const float> ranges, gather_policy policy)
{
    auto& data = *state;

    if (data.failed || ranges.size() != data.table.ray_count || !valid_gather_policy(policy))
        return std::unexpected(operation_error{"invalid sensor update or failed CUDA field"});
    data.ready = false;
    auto status = data.check(cudaSetDevice(data.device));
    if (status) status = data.check(cudaEventRecord(data.events[0], data.stream));
    if (status) status = data.check(cudaMemcpyAsync(data.ranges, ranges.data(), ranges.size_bytes(), cudaMemcpyHostToDevice, data.stream));
    if (status) status = data.check(cudaEventRecord(data.events[1], data.stream));
    if (status) status = data.check(data.options.backend == gather_backend::csr
        ? cuda_sensor_gather(data.table, data.ranges, data.output, policy, data.stream)
        : cuda_sensor_prepared(data.layout, data.ranges, data.prepared, data.output, data.table.settings,
            data.table.ray_count, data.options.backend, policy, data.stream,
            data.bin_events[0] ? data.bin_events.data() : nullptr));
    if (status) status = data.check(cudaEventRecord(data.events[2], data.stream));
    const auto waited = data.check(cudaStreamSynchronize(data.stream));
    if (status) status = waited;
    update_times result;
    if (status) status = data.check(cudaEventElapsedTime(&result.upload_ms, data.events[0], data.events[1]));
    if (status) status = data.check(cudaEventElapsedTime(&result.gather_ms, data.events[1], data.events[2]));
    if (status && data.bin_events[0]) {
        status = data.check(cudaEventElapsedTime(&result.prepare_ms, data.events[1], data.bin_events[0]));

        for (unsigned bin = 0; bin < 4 && status; ++bin)
            status = data.check(cudaEventElapsedTime(&result.bin_ms[bin], data.bin_events[bin], data.bin_events[bin + 1]));
    }
    if (!status) return std::unexpected(status.error());
     data.ready = true;
    return result;
}
operation_result<std::vector<voxel>> gpu_grid::download()
{
    auto& data = *state;

    if (!data.ready || data.failed)
        return std::unexpected(operation_error{"sensor field has no completed update"});
    std::vector<voxel> result(data.grid.voxels);
    auto status = data.check(cudaSetDevice(data.device));
    if (status) status = data.check(cudaMemcpyAsync(result.data(), data.output,
        result.size() * sizeof(voxel), cudaMemcpyDeviceToHost, data.stream));
    const auto waited = data.check(cudaStreamSynchronize(data.stream));
    if (status) status = waited;
    if (!status) return std::unexpected(status.error());
    return result;
}
operation_result<std::unique_ptr<gpu_grid>> gpu_grid::create(const lookup& table,
    int device, std::size_t maximum_bytes, gather_options options)
{
    auto status = validate_lookup(table);
    if (!status) return std::unexpected(status.error());
    if (!safe_accumulation(table.settings, table.ray_count))
        return std::unexpected(operation_error{"sensor accumulation exceeds selected precision range"});
    if (unsigned(options.backend) > 4 || !options.short_list_max)
        return std::unexpected(operation_error{"invalid gather backend or row threshold"});
    std::size_t bytes = table.offsets.size() * 4 + table.intersections.size() * sizeof(intersection) +
        table.active_voxels.size() * 4 + std::size_t(table.ray_count) * 4 + std::size_t(table.grid.voxels) * sizeof(voxel);
    prepared_layout layout;
    row_plan plan;
    subwarp_plan groups;
    if (options.backend != gather_backend::csr) {
        auto built = build_prepared_layout(table, options.backend == gather_backend::thread_interleaved_prepared,
            std::min<std::size_t>(67108864, maximum_bytes / 12));
        if (!built) return std::unexpected(built.error());
        layout = std::move(*built);
        bytes += layout.cells.size() * 12 + layout.ray_ids.size() * 12 + std::size_t(table.ray_count) * sizeof(prepared_ray);
        if (options.backend == gather_backend::warp_soa_hybrid_prepared)
        {
            auto classified = classify_rows(layout.lengths, options.short_list_max);
            if (!classified) return std::unexpected(classified.error());
            plan = std::move(*classified);
            bytes += plan.rows.size() * sizeof(std::uint32_t);
        }
        if (options.backend == gather_backend::warp_soa_subwarp_prepared) {
            auto classified = classify_subwarps(layout.lengths);
            if (!classified) return std::unexpected(classified.error());
            groups = std::move(*classified);
            bytes += groups.rows.size() * sizeof(std::uint32_t);
        }
    }

    if (device < 0 || bytes > maximum_bytes)
        return std::unexpected(operation_error{"sensor field exceeds device storage budget"});
    auto owner = std::unique_ptr<gpu_grid>(new gpu_grid);
    owner->state = std::make_unique<implementation>();
    auto& data = *owner->state;
    data.device = device;
    data.options = options;
    data.grid = table.grid;
     data.table = host_view(table);
    data.bytes = bytes;
    status = data.check(cudaSetDevice(device));
    if (!status) return std::unexpected(status.error());
    data.selected = true;
    status = data.check(cudaStreamCreateWithFlags(&data.stream, cudaStreamNonBlocking));
    std::uint32_t* offsets{}, *active{};
    intersection* entries{};
    if (status) status = data.allocate(offsets, table.offsets.size());
    if (status) status = data.allocate(entries, table.intersections.size());
    if (status) status = data.allocate(active, table.active_voxels.size());
    if (status) status = data.allocate(data.ranges, table.ray_count);
    if (status) status = data.allocate(data.output, table.grid.voxels);
    if (options.backend != gather_backend::csr) {
        data.layout.rows = static_cast<std::uint32_t>(layout.cells.size());
        data.layout.entries = static_cast<std::uint32_t>(layout.ray_ids.size());
        data.layout.stride = layout.stride;
        if (status) status = data.allocate(data.prepared, table.ray_count);
        if (status) status = data.upload(layout.starts, data.layout.starts);
        if (status) status = data.upload(layout.lengths, data.layout.lengths);
        if (status) status = data.upload(layout.cells, data.layout.cells);
        if (status) status = data.upload(layout.ray_ids, data.layout.ray_ids);
        if (status) status = data.upload(layout.entry_t, data.layout.entry_t);
        if (status) status = data.upload(layout.projected, data.layout.projected);
        if (!plan.rows.empty()) {
            data.layout.short_rows = plan.short_rows;
            if (status) status = data.upload(plan.rows, data.layout.row_ids);
        }
        if (!groups.rows.empty()) {
            for (unsigned i = 0; i < 2; ++i) std::copy_n(groups.bin_ends[i], 4, data.layout.bin_ends[i]);
            if (status) status = data.upload(groups.rows, data.layout.row_ids);
        }
    }
    data.table.offsets = offsets;
    data.table.intersections = entries;
    data.table.active_voxels = active;

    for (auto& event : data.events)
        if (status) status = data.check(cudaEventCreate(&event));
    if (options.backend == gather_backend::warp_soa_subwarp_prepared)
        for (auto& event : data.bin_events)
            if (status) status = data.check(cudaEventCreate(&event));
    if (status) status = data.check(cudaMemcpyAsync(offsets, table.offsets.data(), table.offsets.size() * 4, cudaMemcpyHostToDevice, data.stream));
    if (status) status = data.check(cudaMemcpyAsync(entries, table.intersections.data(), table.intersections.size() * sizeof(intersection), cudaMemcpyHostToDevice, data.stream));
    if (status) status = data.check(cudaMemcpyAsync(active, table.active_voxels.data(), table.active_voxels.size() * 4, cudaMemcpyHostToDevice, data.stream));
    if (status) status = data.check(cudaMemsetAsync(data.output, 0, std::size_t(table.grid.voxels) * sizeof(voxel), data.stream));
    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (!status) return std::unexpected(status.error());
    return owner;
}
}
