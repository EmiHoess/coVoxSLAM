#pragma once
#include <covoxslam/storage/nested_array.hpp>
#include <covoxslam/storage/cuda_draft.hpp>

#if defined(covoxslam_has_urd)
namespace covoxslam::storage
{
template<schema s, std::size_t lanes, nested_array_policy policy, bool tagged>
class cuda_nested_draft
{
    using host_type = nested_draft<s, lanes, policy, tagged>;
    std::unique_ptr<device_allocation> allocation_;
    nested_descriptor descriptor_;
    operation_status current_device() const
    {
        if (!allocation_ || !allocation_->ready) return std::unexpected(operation_error{"nested CUDA payload is unavailable"});
        int device{-1};
        auto checked = cuda_storage_status(cudaGetDevice(&device));
        if (!checked) return checked;
        if (device != descriptor_.device) return std::unexpected(operation_error{"nested CUDA allocation device is not current"});
        return {};
    }
    operation_status transfer(nested_descriptor host, bool upload)
    {
        auto checked = current_device();
        if (!checked) return checked;
        if (host.size_tiles != descriptor_.size_tiles || host.lanes != lanes ||
            host.group_stride != descriptor_.group_stride || host.tile_stride != descriptor_.tile_stride ||
            host.voxel_stride != descriptor_.voxel_stride || host.member_stride != descriptor_.member_stride ||
            host.capacity_tiles < host.size_tiles || host.capacity_tiles % lanes)
            return std::unexpected(operation_error{"nested transfer geometry differs"});
        if (upload && descriptor_.generation == std::numeric_limits<std::uint64_t>::max())
            return std::unexpected(operation_error{"nested CUDA generation exhausted"});
        const auto groups = (host.size_tiles + lanes - 1) / lanes;
        if (upload) { allocation_->ready = false; ++descriptor_.generation; }
        cudaError_t submitted = cudaSuccess;

        for (std::size_t group = 0; group < groups && submitted == cudaSuccess; ++group)
            for (std::size_t field = 0; field < field_count<s> && submitted == cudaSuccess; ++field)
            {
                auto* cpu = reinterpret_cast<std::byte*>(std::uintptr_t(host.fields[field]) + group * host.group_stride);
                auto* gpu = descriptor_.fields[field] + group * descriptor_.group_stride;
                submitted = cudaMemcpy2DAsync(upload ? gpu : cpu, descriptor_.member_stride,
                    upload ? cpu : gpu, descriptor_.member_stride, 4, lanes * 512,
                    upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, allocation_->stream);
            }
        const auto complete = cudaStreamSynchronize(allocation_->stream);
        if (submitted != cudaSuccess) { allocation_->ready = false; return cuda_storage_status(submitted); }
        if (complete != cudaSuccess) { allocation_->ready = false; return cuda_storage_status(complete); }
        allocation_->ready = true;
        return {};
    }
public:
    cuda_nested_draft() = default;
    cuda_nested_draft(const cuda_nested_draft&) = delete;
    cuda_nested_draft& operator=(const cuda_nested_draft&) = delete;
    operation_status initialize(const host_type& source)
    {
        if (allocation_ || !source.view().valid()) return std::unexpected(operation_error{"invalid nested CUDA initialization"});
        auto replacement = std::make_unique<device_allocation>();
        auto geometry = source.descriptor();
        auto checked = cuda_storage_status(cudaGetDevice(&geometry.device));
        if (!checked) return checked;
        replacement->descriptor.device = geometry.device;
        checked = cuda_storage_status(cudaStreamCreateWithFlags(&replacement->stream, cudaStreamNonBlocking));
        if (!checked) return checked;
        if (source.allocated_bytes()) {
            checked = cuda_storage_status(cudaMalloc(reinterpret_cast<void**>(&replacement->allocations[0]), source.allocated_bytes()));
            if (!checked) return checked;
            auto first = std::uintptr_t(geometry.fields[0]);

            for (std::size_t field = 1; field < field_count<s>; ++field) first = std::min(first, std::uintptr_t(geometry.fields[field]));

            for (std::size_t field = 0; field < field_count<s>; ++field)
                geometry.fields[field] = replacement->allocations[0] + (std::uintptr_t(geometry.fields[field]) - first);
        }
        geometry.generation = 1;
        checked = cuda_storage_status(static_cast<cudaError_t>(initialize_nested_fields(geometry, replacement->stream)));
        if (!checked) return checked;
        checked = cuda_storage_status(cudaStreamSynchronize(replacement->stream));
        if (!checked) return checked;
        replacement->ready = true;
        allocation_ = std::move(replacement);
        descriptor_ = geometry;
        return {};
    }
    operation_status upload(const host_type& source)
    {
        if (!source.view().valid()) return std::unexpected(operation_error{"nested upload source is uninitialized"});
        return transfer(source.descriptor(), true);
    }
    operation_status download(host_type& destination)
    {
        if (!destination.view().valid()) return std::unexpected(operation_error{"nested download destination is uninitialized"});
        return transfer(destination.descriptor(), false);
    }
    operation_result<nested_descriptor> descriptor() const
    {
        auto checked = current_device();
        if (!checked) return std::unexpected(checked.error());
        return descriptor_;
    }
    operation_result<map_view<s, nested_device_accessor<s>>> view()
    {
        auto current = descriptor();
        if (!current) return std::unexpected(current.error());
        return map_view<s, nested_device_accessor<s>>{nested_device_accessor<s>{*current}};
    }
    cudaStream_t stream() const noexcept { return allocation_ ? allocation_->stream : nullptr; }
};
}
#endif
