#include <covoxslam/store/device_index.hpp>
#include <cuda_runtime.h>
#include <new>

namespace covoxslam::store
{
__global__ void initialize_index(device_hash_entry* entries, std::size_t capacity)
{
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < capacity) ::new (static_cast<void*>(entries + index)) device_hash_entry{};
}
__global__ void validate_index(const device_hash_entry* entries, std::size_t capacity, unsigned* errors)
{
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= capacity || entries[index].state == 0) return;
    const auto& entry = entries[index];
    if (entry.state != 1 || !entry.reuse_generation) { atomicAdd(errors, 1U); return; }
    auto bucket = device_key_hash(entry.key) & (capacity - 1);

    for (std::size_t probe = 0; probe < capacity; ++probe) {
        if (entries[bucket].state == 1 && device_key_equal(entries[bucket].key, entry.key)) {
            if (bucket != index) atomicAdd(errors, 1U);
            return;
        }
        if (!entries[bucket].state) break;
        bucket = (bucket + 1) & (capacity - 1);
    }
    atomicAdd(errors, 1U);
}
__global__ void populate_index(device_hash_entry* entries, std::size_t capacity,
    const device_hash_entry* records, std::size_t count, unsigned* errors)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto record = records[i];
    auto bucket = device_key_hash(record.key) & (capacity - 1);

    for (std::size_t probe = 0; probe < capacity; ++probe) {
        auto& target = entries[bucket];
        if (atomicCAS(&target.state, 0U, 2U) == 0U) {
            target.key = record.key;
            target.slot = record.slot;
            target.reuse_generation = record.reuse_generation;
            __threadfence();
            atomicExch(&target.state, 1U);
            return;
        }
        bucket = (bucket + 1) & (capacity - 1);
    }
    atomicAdd(errors, 1U);
}
int build_device_index(device_hash_entry* entries, std::size_t capacity, const device_hash_entry* records,
    std::size_t count, unsigned* errors, void* stream) noexcept
{
    if (!entries || !errors || !capacity || (capacity & (capacity - 1)) || count > capacity / 2 || (count && !records))
        return int(cudaErrorInvalidValue);
    if (capacity > std::size_t(1) << 31) return int(cudaErrorInvalidValue);
    const auto queue = static_cast<cudaStream_t>(stream);
    initialize_index<<<unsigned((capacity + 127) / 128), 128, 0, queue>>>(entries, capacity);
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    if (!count) return int(cudaSuccess);
    populate_index<<<unsigned((count + 127) / 128), 128, 0, queue>>>(entries, capacity, records, count, errors);
    error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    validate_index<<<unsigned((capacity + 127) / 128), 128, 0, queue>>>(entries, capacity, errors);
    return int(cudaGetLastError());
}
}
