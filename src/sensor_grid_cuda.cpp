#include <covoxslam/sensor_grid_cuda.hpp>
#include <covoxslam/sensor_grid_precision.h>
#include <covoxslam/sensor_grid_layout.hpp>
#include "sensor_grid/native.h"
#include <cuda_runtime_api.h>
#include <array>
#include <algorithm>
#include <chrono>
#include "sensor_grid_graph.hpp"

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
    compact_ray* compact{};
    float* ranges{};
     voxel* output{};
    cudaStream_t stream{};
    std::array<cudaEvent_t, 3> events{};
    std::array<cudaEvent_t, 5> bin_events{};
    std::array<void*, 14> allocations{};
    std::size_t count{}, bytes{};
    bool selected{}, failed{}, ready{};
    std::array<graph_plan, 4> graphs{};
    std::uint64_t generation{1};
    unsigned next_graph{};
    execution_statistics execution_stats{};
    int launch(gather_policy policy)
    {
        switch (options.backend)
        {
        case gather_backend::csr: return cuda_sensor_gather(table, ranges, output, policy, stream);
        case gather_backend::hybrid_compact:
            return cuda_sensor_compact(layout, ranges, compact, output, table.settings, table.ray_count, policy, stream);
        case gather_backend::hybrid_pipelined:
            return cuda_sensor_pipelined(layout, ranges, prepared, output, table.settings, table.ray_count, policy, stream);
        case gather_backend::hybrid_block_long:
            return cuda_sensor_block_long(layout, ranges, prepared, output, table.settings, table.ray_count, policy, stream);
        default: return cuda_sensor_prepared(layout, ranges, prepared, output, table.settings, table.ray_count,
            options.backend, policy, stream, bin_events[0] ? bin_events.data() : nullptr, options.execution == execution_mode::cuda_graph);
        }
    }
    operation_result<cudaGraphExec_t> prepare_graph(gather_policy policy)
    {
        if (options.execution == execution_mode::stream) return nullptr;
        const graph_key key{options.backend, policy.max_rays, accumulation_bits, generation,
            policy.first_entry_only, bin_events[0] != nullptr,
            {ranges, output, prepared ? static_cast<void*>(prepared) : static_cast<void*>(compact), table.offsets, layout.row_ids}};

        for (auto& plan : graphs)
            if (plan.executable && plan.key == key) { ++execution_stats.graph_hits; return plan.executable; }
        const auto start = std::chrono::steady_clock::now();
        graph_plan plan;
        plan.key = key;
        auto status = check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        if (!status) return std::unexpected(status.error());
        status = check(launch(policy));
        const auto ended = check(cudaStreamEndCapture(stream, &plan.graph));
        if (status) status = ended;
        if (status) status = check(cudaGraphInstantiate(&plan.executable, plan.graph, 0));
        if (status) status = check(cudaGraphUpload(plan.executable, stream));
        const auto waited = check(cudaStreamSynchronize(stream));
        if (status) status = waited;

        if (!status) {
            plan.release();
            return std::unexpected(status.error());
        }
        auto& slot = graphs[next_graph++ % graphs.size()];
        slot.release();
        slot = plan;
        ++execution_stats.graph_builds;
        execution_stats.setup_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        return slot.executable;
    }
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

        for (auto& plan : graphs) plan.release();

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
execution_statistics gpu_grid::execution_statistics_value() const noexcept { return state->execution_stats; }
gather_statistics gpu_grid::statistics() const noexcept
{
    const auto& view = state->layout;
    return {view.short_rows, view.rows - view.short_rows - view.block_rows, view.block_rows,
        state->table.entry_count, view.entries ? view.entries : state->table.entry_count};
}
operation_status gpu_grid::invalidate_graphs()
{
    auto& data = *state;
    const auto start = std::chrono::steady_clock::now();
    if (data.failed || data.generation == UINT64_MAX)
        return std::unexpected(operation_error{"cannot invalidate failed sensor graph generation"});
    auto status = data.check(cudaSetDevice(data.device));
    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (!status) return status;

    for (auto& plan : data.graphs) plan.release();
    ++data.generation;
    data.next_graph = 0;
    ++data.execution_stats.invalidations;
    data.execution_stats.invalidation_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
     return {};
}
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
    if (!status) return std::unexpected(status.error());
    const auto previous = data.execution_stats;
    const auto graph = data.prepare_graph(policy);
    if (!graph) return std::unexpected(graph.error());
    if (status) status = data.check(cudaEventRecord(data.events[0], data.stream));
    if (status) status = data.check(cudaMemcpyAsync(data.ranges, ranges.data(), ranges.size_bytes(), cudaMemcpyHostToDevice, data.stream));
    if (status) status = data.check(cudaEventRecord(data.events[1], data.stream));
    if (status) status = data.check(*graph ? int(cudaGraphLaunch(*graph, data.stream)) : data.launch(policy));
    if (status) status = data.check(cudaEventRecord(data.events[2], data.stream));
    const auto waited = data.check(cudaStreamSynchronize(data.stream));
    if (status) status = waited;
    update_times result;
    result.graph_setup_ms = data.execution_stats.setup_ms - previous.setup_ms;
    result.graph_reused = data.execution_stats.graph_hits != previous.graph_hits;
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
    if (unsigned(options.backend) > 10 || !options.short_list_max || unsigned(options.execution) > 1)
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
        if (options.backend >= gather_backend::hybrid_short_interleaved && options.backend <= gather_backend::hybrid_ray16) {
            auto packed = build_hybrid_layout(table, options.backend, options.short_list_max,
                std::min<std::size_t>(67108864, maximum_bytes / 12));
            if (!packed) return std::unexpected(packed.error());
            layout = std::move(packed->data);
            plan = std::move(packed->plan);
            bytes += layout.ranges.size() * sizeof(row_range) + plan.rows.size() * 4;
        }
        bytes += layout.cells.size() * 12 + layout.ray_ids.size() * 12 + std::size_t(table.ray_count) *
            (options.backend == gather_backend::hybrid_compact ? sizeof(compact_ray) : sizeof(prepared_ray));
        bytes -= layout.ray_ids16.size() * 2;
        if (options.backend == gather_backend::warp_soa_hybrid_prepared || options.backend == gather_backend::hybrid_compact ||
            options.backend == gather_backend::hybrid_pipelined)
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
        if (options.backend == gather_backend::hybrid_block_long) {
            auto classified = classify_blocks(layout.lengths, options.short_list_max, options.block_list_min);
            if (!classified) return std::unexpected(classified.error());
            plan = std::move(*classified);
            bytes += plan.rows.size() * 4;
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
        if (status) status = options.backend == gather_backend::hybrid_compact
            ? data.allocate(data.compact, table.ray_count) : data.allocate(data.prepared, table.ray_count);
        if (status) status = data.upload(layout.starts, data.layout.starts);
        if (status) status = data.upload(layout.lengths, data.layout.lengths);
        if (status) status = data.upload(layout.cells, data.layout.cells);
        if (status) status = layout.ray_ids16.empty() ? data.upload(layout.ray_ids, data.layout.ray_ids)
            : data.upload(layout.ray_ids16, data.layout.ray_ids16);
        if (status) status = data.upload(layout.entry_t, data.layout.entry_t);
        if (status) status = data.upload(layout.projected, data.layout.projected);
        data.layout.short_stride = layout.short_stride;
        if (status && !layout.ranges.empty()) status = data.upload(layout.ranges, data.layout.ranges);
        if (!plan.rows.empty()) {
            data.layout.short_rows = plan.short_rows;
            data.layout.block_rows = plan.block_rows;
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
