#include <covoxslam/sensor_grid_cuda.hpp>
#include <covoxslam/sensor_grid_precision.h>
#include "sensor_grid/native.h"
#include <cuda_runtime_api.h>
#include <array>

namespace covoxslam::sensor_grid
{
struct gpu_grid::implementation
{
    int device{};
    geometry grid{};
    lookup_view table{};
    float* ranges{};
     voxel* output{};
    cudaStream_t stream{};
    std::array<cudaEvent_t, 3> events{};
    std::array<void*, 5> allocations{};
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
    if (status) status = data.check(cuda_sensor_gather(data.table, data.ranges, data.output, policy, data.stream));
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
    int device, std::size_t maximum_bytes)
{
    auto status = validate_lookup(table);
    if (!status) return std::unexpected(status.error());
    if (!safe_accumulation(table.settings, table.ray_count))
        return std::unexpected(operation_error{"sensor accumulation exceeds selected precision range"});
    const std::size_t bytes = table.offsets.size() * 4 + table.intersections.size() * sizeof(intersection) +
        table.active_voxels.size() * 4 + std::size_t(table.ray_count) * 4 + std::size_t(table.grid.voxels) * sizeof(voxel);

    if (device < 0 || bytes > maximum_bytes)
        return std::unexpected(operation_error{"sensor field exceeds device storage budget"});
    auto owner = std::unique_ptr<gpu_grid>(new gpu_grid);
    owner->state = std::make_unique<implementation>();
    auto& data = *owner->state;
    data.device = device;
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
    data.table.offsets = offsets;
    data.table.intersections = entries;
    data.table.active_voxels = active;

    for (auto& event : data.events)
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
