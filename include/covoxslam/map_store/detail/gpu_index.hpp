#pragma once
#include <covoxslam/core/byte_units.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/device_index.hpp>
#include <covoxslam/map_store/cpu_index.hpp>
#include <covoxslam/storage/device_buffer.hpp>

namespace covoxslam::store
{
struct index_allocation
{
    device_hash_entry* entries{};
    device_hash_entry* input{};
    unsigned* errors{};
    cudaStream_t stream{};
    device_index_view descriptor;
    ~index_allocation()
    {
        int previous{-1};
        const auto queried = cudaGetDevice(&previous);
        if (descriptor.device < 0 || cudaSetDevice(descriptor.device) != cudaSuccess) return;
        if (stream) cudaStreamSynchronize(stream);
        if (entries) cudaFree(entries);
        if (input) cudaFree(input);
        if (errors) cudaFree(errors);
        if (stream) cudaStreamDestroy(stream);
        if (queried == cudaSuccess && previous != descriptor.device) cudaSetDevice(previous);
    }
};
class gpu_block_index
{
    std::shared_ptr<index_allocation> state_;
public:
    operation_status initialize(const cpu_block_index& source, std::uint64_t allocation_generation,
        std::size_t byte_budget = 256U * bytes_per_mebibyte)
    {
        if (state_ || !source.initialized() || !allocation_generation)
            return std::unexpected(operation_error{"invalid GPU index initialization"});
        std::size_t capacity{2};
        if (source.size() > std::size_t(std::numeric_limits<std::uint32_t>::max()) / 2)
            return std::unexpected(operation_error{"GPU index exceeds supported capacity"});
        while (capacity / 2 < source.size()) capacity *= 2;
        const auto bytes = (capacity + source.size()) * sizeof(device_hash_entry) + sizeof(unsigned);
        if (bytes > byte_budget) return std::unexpected(operation_error{"GPU index exceeds byte budget"});
        std::vector<device_hash_entry> records;
        records.reserve(source.size());

        for (std::size_t slot = 0; slot < source.records().size(); ++slot) {
            const auto& record = source.records()[slot];
            if (record.occupied) records.push_back({record.key, record.generation, std::uint32_t(slot), 1});
        }
        auto candidate = std::make_shared<index_allocation>();
        auto error = cudaGetDevice(&candidate->descriptor.device);
        if (error == cudaSuccess) error = cudaStreamCreateWithFlags(&candidate->stream, cudaStreamNonBlocking);
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void**>(&candidate->entries), capacity * sizeof(device_hash_entry));
        if (error == cudaSuccess && !records.empty()) error = cudaMalloc(reinterpret_cast<void**>(&candidate->input), records.size() * sizeof(device_hash_entry));
        if (error == cudaSuccess) error = cudaMalloc(reinterpret_cast<void**>(&candidate->errors), sizeof(unsigned));
        if (error != cudaSuccess) return storage::cuda_storage_status(error);
        error = cudaMemsetAsync(candidate->errors, 0, sizeof(unsigned), candidate->stream);
        if (error == cudaSuccess && !records.empty()) error = cudaMemcpyAsync(candidate->input, records.data(),
            records.size() * sizeof(device_hash_entry), cudaMemcpyHostToDevice, candidate->stream);
        if (error == cudaSuccess) error = cudaError_t(build_device_index(candidate->entries, capacity,
            candidate->input, records.size(), candidate->errors, candidate->stream));
        unsigned failures{};
        if (error == cudaSuccess) error = cudaMemcpyAsync(&failures, candidate->errors, sizeof(failures), cudaMemcpyDeviceToHost, candidate->stream);
        const auto completion = cudaStreamSynchronize(candidate->stream);
        if (error != cudaSuccess) return storage::cuda_storage_status(error);
        if (completion != cudaSuccess) return storage::cuda_storage_status(completion);
        if (failures) return std::unexpected(operation_error{"GPU index insertion failed"});
        auto* input = std::exchange(candidate->input, nullptr);
        if (input) {
            error = cudaFree(input);
            if (error != cudaSuccess) return storage::cuda_storage_status(error);
        }
        auto* errors = std::exchange(candidate->errors, nullptr);
        error = cudaFree(errors);
        if (error != cudaSuccess) return storage::cuda_storage_status(error);
        candidate->descriptor.entries = candidate->entries;
        candidate->descriptor.capacity = capacity;
        candidate->descriptor.submap = source.submap();
        candidate->descriptor.field = std::uint32_t(source.field());
        candidate->descriptor.allocation_generation = allocation_generation;
        state_ = std::move(candidate);
        return {};
    }
    operation_result<device_index_view> view() const
    {
        if (!state_) return std::unexpected(operation_error{"GPU index is not initialized"});
        int device{-1};
        const auto checked = storage::cuda_storage_status(cudaGetDevice(&device));
        if (!checked) return std::unexpected(checked.error());
        if (device != state_->descriptor.device) return std::unexpected(operation_error{"GPU index requires its device"});
        return state_->descriptor;
    }
};
}
