#include <covoxslam/store/device_index.hpp>
#include <cuda_runtime.h>

namespace covoxslam::store
{
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
    if (!count) return int(cudaSuccess);
    populate_index<<<unsigned((count + 127) / 128), 128, 0, static_cast<cudaStream_t>(stream)>>>(entries, capacity, records, count, errors);
    return int(cudaGetLastError());
}
}
