#pragma once
#include <covoxslam/storage/draft.hpp>
#include <covoxslam/storage/device_view.hpp>
#include <cuda_runtime_api.h>
#include <span>

namespace covoxslam::storage
{
inline operation_status cuda_storage_status(cudaError_t status)
{
    if (status != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(status)});
    return {};
}
struct device_allocation
{
    std::byte* allocations[5]{};
    device_descriptor descriptor;
    cudaStream_t stream{};
    bool ready{};
    ~device_allocation()
    {
        int previous{-1};
        const auto queried = cudaGetDevice(&previous);
        if (descriptor.device < 0 || cudaSetDevice(descriptor.device) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);

        for (auto allocation : allocations) if (allocation) cudaFree(allocation);
        if (stream) cudaStreamDestroy(stream);
        if (queried == cudaSuccess && previous != descriptor.device) cudaSetDevice(previous);
    }
};
template<schema s, layout l> class cuda_draft
{
    using traits = storage_traits<s, l>;
    std::unique_ptr<device_allocation> state_;
    operation_status require_device() const
    {
        if (!state_) return std::unexpected(operation_error{"CUDA collection is uninitialized"});
        int current{-1};
        auto queried = cuda_storage_status(cudaGetDevice(&current));

        if (!queried) return queried;
        if (current != state_->descriptor.device)
            return std::unexpected(operation_error{"CUDA collection requires its allocation device"});
        return {};
    }
public:
    cuda_draft() = default;
    cuda_draft(const cuda_draft&) = delete;
    cuda_draft& operator=(const cuda_draft&) = delete;
    operation_status initialize(std::size_t count);
    operation_result<device_map_view<s>> view();
    operation_result<device_map_view<s, true>> view() const;
    template<layout host_layout> operation_status upload(const draft<s, host_layout>& source);
    template<layout host_layout> operation_status download(draft<s, host_layout>& destination) const;
    cudaStream_t stream() const noexcept { return state_ ? state_->stream : nullptr; }
    std::size_t size() const noexcept { return state_ ? state_->descriptor.size_tiles : 0; }
    operation_status upload_direct(std::span<const aos_tile<s>> source) requires (l == layout::pooled_aos)
    {
        auto checked = require_device();
        if (!checked) return checked;
        if (source.size() != size() || state_->descriptor.generation == std::numeric_limits<std::uint64_t>::max())
            return std::unexpected(operation_error{"CUDA direct upload size or generation is invalid"});
        state_->ready = false;
        ++state_->descriptor.generation;
        const auto submitted = source.empty() ? cudaSuccess : cudaMemcpyAsync(state_->allocations[0], source.data(),
            source.size_bytes(), cudaMemcpyHostToDevice, state_->stream);
        const auto completed = cudaStreamSynchronize(state_->stream);

        if (submitted != cudaSuccess) return cuda_storage_status(submitted);
        if (completed != cudaSuccess) return cuda_storage_status(completed);
        state_->ready = true;
        return {};
    }
    operation_result<device_descriptor> descriptor() const
    {
        auto checked = require_device();

        if (!checked) return std::unexpected(checked.error());
        if (!state_->ready) return std::unexpected(operation_error{"CUDA collection payload is unavailable"});
        return state_->descriptor;
    }
    bool is_current(const device_descriptor& candidate) const
    {
        auto current = descriptor();
        if (!current || candidate.generation != current->generation || candidate.device != current->device ||
            candidate.record_schema != s || candidate.storage_layout != l || candidate.size_tiles != size() ||
            candidate.capacity_tiles != current->capacity_tiles || candidate.tile_stride != traits::tile_stride ||
            candidate.voxel_stride != traits::voxel_stride) return false;

        for (std::size_t i = 0; i < field_count<s>; ++i)
            if (candidate.fields[i] != current->fields[i]) return false;
        return true;
    }
    operation_status complete()
    {
        auto checked = require_device();
        if (!checked) return checked;
        checked = cuda_storage_status(cudaStreamSynchronize(state_->stream));
        if (!checked) state_->ready = false;
        return checked;
    }
};
template<schema s, layout l>
operation_status cuda_draft<s, l>::initialize(std::size_t count)
{
    if (state_) return std::unexpected(operation_error{"CUDA collection is already initialized"});
    constexpr std::size_t arrays = l == layout::pool_soa ? field_count<s> : 1;
    constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());

    if (count > maximum / traits::tile_stride / arrays)
        return std::unexpected(operation_error{"CUDA collection capacity overflow"});
    auto replacement = std::make_unique<device_allocation>();
    auto& descriptor = replacement->descriptor;
    auto checked = cuda_storage_status(cudaGetDevice(&descriptor.device));
    if (!checked) return checked;
    checked = cuda_storage_status(cudaStreamCreateWithFlags(&replacement->stream, cudaStreamNonBlocking));
    if (!checked) return checked;
    descriptor.record_schema = s;
    descriptor.storage_layout = l;
    descriptor.size_tiles = descriptor.capacity_tiles = count;
    descriptor.tile_stride = traits::tile_stride;
    descriptor.voxel_stride = traits::voxel_stride;
    descriptor.generation = 1;

    for (std::size_t i = 0; i < arrays && count != 0; ++i) {
        checked = cuda_storage_status(cudaMalloc(reinterpret_cast<void**>(&replacement->allocations[i]), count * traits::tile_stride));
        if (!checked) return checked;
    }

    for (std::size_t field = 0; field < field_count<s>; ++field)
    {
        if constexpr (l == layout::pool_soa) descriptor.fields[field] = replacement->allocations[field];
        else if (count) descriptor.fields[field] = reinterpret_cast<std::byte*>(
            reinterpret_cast<std::uintptr_t>(replacement->allocations[0]) + traits::offsets[field]);
    }
    checked = cuda_storage_status(static_cast<cudaError_t>(initialize_device_fields(descriptor, replacement->stream)));
    if (!checked) return checked;
    checked = cuda_storage_status(cudaStreamSynchronize(replacement->stream));
    if (!checked) return checked;
    replacement->ready = true;
    state_ = std::move(replacement);
    return {};
}
template<schema s, layout l> operation_result<device_map_view<s>> cuda_draft<s, l>::view()
{
    auto checked = require_device();
    if (!checked) return std::unexpected(checked.error());
    if (!state_ || !state_->ready) return std::unexpected(operation_error{"CUDA collection is not ready"});
    return device_map_view<s>{device_accessor<s>{state_->descriptor}};
}
template<schema s, layout l> operation_result<device_map_view<s, true>> cuda_draft<s, l>::view() const
{
    auto checked = require_device();
    if (!checked) return std::unexpected(checked.error());
    if (!state_ || !state_->ready) return std::unexpected(operation_error{"CUDA collection is not ready"});
    return device_map_view<s, true>{device_accessor<s, true>{state_->descriptor}};
}
template<schema s, layout l> template<layout host_layout>
operation_status cuda_draft<s, l>::upload(const draft<s, host_layout>& source)
{
    auto checked = require_device();
    if (!checked) return checked;
    if (!state_ || source.size() != size())
        return std::unexpected(operation_error{"CUDA upload collection size mismatch"});
    if (state_->descriptor.generation == std::numeric_limits<std::uint64_t>::max())
        return std::unexpected(operation_error{"CUDA collection generation exhausted"});
    ++state_->descriptor.generation;
    state_->ready = false;
    cudaError_t submitted = cudaSuccess;

    for (std::size_t tile = 0; tile < size() && submitted == cudaSuccess; ++tile)
    {
        [&]<std::size_t... field>(std::index_sequence<field...>) {
            const void* fields[]{&source.template scalar<field>(tile, 0)...};

            for (std::size_t i = 0; i < field_count<s> && submitted == cudaSuccess; ++i)
                submitted = cudaMemcpy2DAsync(reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(state_->descriptor.fields[i]) +
                    tile * traits::tile_stride), traits::voxel_stride, fields[i], storage_traits<s, host_layout>::voxel_stride,
                    4, voxels_per_tile, cudaMemcpyHostToDevice, state_->stream);
        }(std::make_index_sequence<field_count<s>>{});
    }
    const auto completed = cudaStreamSynchronize(state_->stream);
    if (submitted != cudaSuccess) return cuda_storage_status(submitted);
    if (completed != cudaSuccess) return cuda_storage_status(completed);
    state_->ready = true;
    return {};
}
template<schema s, layout l> template<layout host_layout>
operation_status cuda_draft<s, l>::download(draft<s, host_layout>& destination) const
{
    auto checked = require_device();
    if (!checked) return checked;
    if (!state_ || !state_->ready || destination.size() != size())
        return std::unexpected(operation_error{"CUDA download collection size mismatch or unavailable payload"});
    cudaError_t submitted = cudaSuccess;

    for (std::size_t tile = 0; tile < size() && submitted == cudaSuccess; ++tile)
    {
        [&]<std::size_t... field>(std::index_sequence<field...>) {
            void* fields[]{&destination.template scalar<field>(tile, 0)...};

            for (std::size_t i = 0; i < field_count<s> && submitted == cudaSuccess; ++i)
                submitted = cudaMemcpy2DAsync(fields[i], storage_traits<s, host_layout>::voxel_stride,
                    reinterpret_cast<const void*>(reinterpret_cast<std::uintptr_t>(state_->descriptor.fields[i]) + tile * traits::tile_stride),
                    traits::voxel_stride, 4, voxels_per_tile, cudaMemcpyDeviceToHost, state_->stream);
        }(std::make_index_sequence<field_count<s>>{});
    }
    const auto completed = cudaStreamSynchronize(state_->stream);
    if (submitted != cudaSuccess) return cuda_storage_status(submitted);
    return cuda_storage_status(completed);
}
}
