#include <covoxslam/sensor/dense_field.hpp>
#include <covoxslam/sensor/detail/precision.hpp>
#include <covoxslam/sensor/limits.hpp>
#include <covoxslam/sensor/prepared_layout.hpp>
#include "mapping/tsdf/gather.cuh"
#include <cuda_runtime_api.h>
#include <array>
#include <algorithm>
#include <chrono>
#include <limits>

namespace covoxslam::sensor_grid
{
namespace
{
operation_result<std::size_t> dense_field_device_bytes(
    const lookup& table, const prepared_layout& layout, const row_plan& plan)
{
    if (layout.ray_ids.size() > maximum_prepared_intersections ||
        (!layout.ray_ids16.empty() && layout.ray_ids16.size() != layout.ray_ids.size()))
        return std::unexpected(operation_error{"prepared sensor layout exceeds capacity"});
    std::size_t total{};
    const auto append = [&total](std::size_t count, std::size_t width) {
        if (count > (std::numeric_limits<std::size_t>::max() - total) / width)
            return false;
        total += count * width;
        return true;
    };
    const auto entry_width = layout.ray_ids16.empty() ?
        sizeof(std::uint32_t) + 2 * sizeof(float) : sizeof(std::uint16_t) + 2 * sizeof(float);
    if (!append(table.ray_count, sizeof(float) + sizeof(compact_ray)) ||
        !append(table.grid.voxels, sizeof(voxel)) ||
        !append(layout.cells.size(), 3 * sizeof(std::uint32_t)) ||
        !append(layout.ray_ids.size(), entry_width) ||
        !append(plan.rows.size(), sizeof(std::uint32_t)))
        return std::unexpected(operation_error{"sensor field device byte count overflows"});
    return total;
}
}

struct gpu_grid::implementation
{
    int device{};
    geometry grid{};
    lookup_view table{};
    layout_view layout{};
    gather_options options{};
    compact_ray* compact{};
    float* ranges{};
     voxel* output{};
    cudaStream_t stream{};
    std::array<cudaEvent_t, 3> events{};
    std::array<void*, 14> allocations{};
    std::size_t count{}, bytes{};
    bool selected{}, failed{}, ready{};
    execution_statistics execution_stats{};
    int launch(gather_policy policy)
    {
        return cuda_sensor_combined(layout, ranges, compact, output, table.settings, table.ray_count, policy, stream);
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


        for (auto event : events)
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
    return std::unexpected(operation_error{"CUDA graphs are not applicable to coVoxSLAM"});
}
field_view gpu_grid::device_field() const noexcept
{
    if (!state->ready || state->failed) return {};
    return {state->output, state->grid.side, state->table.settings.voxel_size, state->grid.origin};
}
operation_result<update_times> gpu_grid::update(std::span<const float> ranges, gather_policy policy)
{
    auto& data = *state;

    if (data.failed || ranges.size() != data.table.ray_count || !valid_gather_policy(policy) ||
        policy.max_rays || policy.first_entry_only)
        return std::unexpected(operation_error{"invalid sensor update or failed CUDA field"});
    data.ready = false;
    auto status = data.check(cudaSetDevice(data.device));
    if (!status) return std::unexpected(status.error());
    if (status) status = data.check(cudaEventRecord(data.events[0], data.stream));
    if (status) status = data.check(cudaMemcpyAsync(data.ranges, ranges.data(), ranges.size_bytes(), cudaMemcpyHostToDevice, data.stream));
    if (status) status = data.check(cudaEventRecord(data.events[1], data.stream));
    if (status) status = data.check(data.launch(policy));
    if (status) status = data.check(cudaEventRecord(data.events[2], data.stream));
    const auto waited = data.check(cudaStreamSynchronize(data.stream));
    if (status) status = waited;
    update_times result;
    if (status) status = data.check(cudaEventElapsedTime(&result.upload_ms, data.events[0], data.events[1]));
    if (status) status = data.check(cudaEventElapsedTime(&result.gather_ms, data.events[1], data.events[2]));
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
    if (options.backend != gather_backend::hybrid_combined || options.execution != execution_mode::stream ||
        options.short_list_max != combined_short_list_max || options.block_list_min != combined_block_list_min)
        return std::unexpected(operation_error{"coVoxSLAM requires combined stream gathering with thresholds 16 and 1024"});
    auto built = build_prepared_layout(table, false,
        std::min<std::size_t>(maximum_prepared_intersections, maximum_bytes / 12));
    if (!built) return std::unexpected(built.error());
    auto layout = std::move(*built);

    if (ray_ids_fit_u16(table.ray_count)) {
        auto packed = pack_ray_ids(layout.ray_ids);
        if (!packed) return std::unexpected(packed.error());
        layout.ray_ids16 = std::move(*packed);
    }
    auto classified = classify_blocks(layout.lengths, options.short_list_max, options.block_list_min);
    if (!classified) return std::unexpected(classified.error());
    const auto plan = std::move(*classified);
    const auto required_bytes = dense_field_device_bytes(table, layout, plan);
    if (!required_bytes) return std::unexpected(required_bytes.error());
    const auto bytes = *required_bytes;

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
    if (status) status = data.allocate(data.ranges, table.ray_count);
    if (status) status = data.allocate(data.output, table.grid.voxels);
    data.layout.rows = static_cast<std::uint32_t>(layout.cells.size());
    data.layout.entries = static_cast<std::uint32_t>(layout.ray_ids.size());
    data.layout.stride = layout.stride;
    if (status) status = data.allocate(data.compact, table.ray_count);
    if (status) status = data.upload(layout.starts, data.layout.starts);
    if (status) status = data.upload(layout.lengths, data.layout.lengths);
    if (status) status = data.upload(layout.cells, data.layout.cells);
    if (status) status = layout.ray_ids16.empty() ? data.upload(layout.ray_ids, data.layout.ray_ids)
        : data.upload(layout.ray_ids16, data.layout.ray_ids16);
    if (status) status = data.upload(layout.entry_t, data.layout.entry_t);
    if (status) status = data.upload(layout.projected, data.layout.projected);
    data.layout.short_rows = plan.short_rows;
    data.layout.block_rows = plan.block_rows;

    if (status) status = data.upload(plan.rows, data.layout.row_ids);

    for (auto& event : data.events)
        if (status) status = data.check(cudaEventCreate(&event));
    if (status) status = data.check(cudaMemsetAsync(data.output, 0, std::size_t(table.grid.voxels) * sizeof(voxel), data.stream));
    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (!status) return std::unexpected(status.error());
    return owner;
}
}
