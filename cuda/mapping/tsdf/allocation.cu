#include "allocation.cuh"
#include "detail/tile_hash.cuh"
#include "../../primitives/radix_sort.cuh"

namespace covoxslam::sensor_grid
{
namespace
{
__device__ unsigned find_resident_key(resident_discovery_view view, std::uint64_t key)
{
    auto bucket = unsigned(detail::resident_hash(key) & (view.scan.hash_capacity - 1));

    for (unsigned probe = 0; probe < view.scan.hash_capacity; ++probe) {

        if (view.hash[bucket].key == key) return view.hash[bucket].slot;
        if (view.hash[bucket].key == UINT64_MAX) return UINT32_MAX;
        bucket = (bucket + 1) & (view.scan.hash_capacity - 1);
    }
    return UINT32_MAX;
}
__device__ void insert_resident_key(resident_discovery_view view, std::uint64_t key, unsigned slot)
{
    auto bucket = unsigned(detail::resident_hash(key) & (view.scan.hash_capacity - 1));

    for (unsigned probe = 0; probe < view.scan.hash_capacity; ++probe) {
        const auto old = atomicCAS(reinterpret_cast<unsigned long long*>(&view.hash[bucket].key), UINT64_MAX, key);

        if (old == UINT64_MAX) { view.hash[bucket].slot = slot; return; }
        if (old == key) { atomicOr(&view.scan.state->error, discovery_ownership); return; }
        bucket = (bucket + 1) & (view.scan.hash_capacity - 1);
    }
    atomicOr(&view.scan.state->error, discovery_hash);
}
__global__ void classify_resident_keys(resident_discovery_view view)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= view.scan.capacity) return;
    view.missing[i] = 0; view.slots[i] = UINT32_MAX;

    if (i >= view.scan.state->count || atomicAdd(&view.scan.state->error, 0U)) return;
    const auto key = view.scan.keys[i];

    if (key == UINT64_MAX || (i && view.scan.keys[i - 1] >= key)) {
        atomicOr(&view.scan.state->error, discovery_ownership); return;
    }
    const auto slot = find_resident_key(view, key);

    if (slot == UINT32_MAX) view.missing[i] = 1;
    else if (slot >= view.scan.state->tile_count) atomicOr(&view.scan.state->error, discovery_ownership);
    else view.slots[i] = slot;
}
__global__ void validate_resident_allocation(resident_discovery_view view)
{
    auto& state = *view.scan.state;
    state.new_count = 0;

    if (state.error) return;
    const auto additional = view.offsets[view.scan.capacity - 1] + view.missing[view.scan.capacity - 1];

    if (state.count > view.scan.capacity || state.tile_count > view.scan.capacity ||
        additional > view.scan.capacity - state.tile_count) {
        state.error |= discovery_pool; return;
    }
    state.new_count = unsigned(additional);
}
__global__ void commit_resident_keys(resident_discovery_view view)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (atomicAdd(&view.scan.state->error, 0U) || i >= view.scan.state->count || !view.missing[i]) return;
    const auto key = view.scan.keys[i];
    const auto slot = view.scan.state->tile_count + unsigned(view.offsets[i]);
    constexpr int bias = 1 << 20;
    view.keys[slot] = {int(key & 2097151) - bias, int((key >> 21) & 2097151) - bias, int(key >> 42) - bias};
    view.slots[i] = slot;
    insert_resident_key(view, key, slot);
}
__global__ void publish_resident_keys(discovery_state* state)
{
    if (!state->error) state->tile_count += state->new_count;
}
__global__ void prepare_resident_restore(discovery_state* state, unsigned count)
{
    *state = {0, 0, 0, count};
}
__global__ void seed_resident_keys(resident_discovery_view view, unsigned count)
{
    const auto slot = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (slot >= count) return;
    const auto coordinate = view.keys[slot];
    const auto key = detail::resident_key({coordinate.z, coordinate.y, coordinate.x});

    if (key == UINT64_MAX) { atomicOr(&view.scan.state->error, discovery_coordinate); return; }
    insert_resident_key(view, key, unsigned(slot));
}
}
int cuda_sensor_restore_tiles(resident_discovery_view view, std::uint32_t count, void* opaque_stream)
{
    if (!valid_resident_discovery(view) || count > view.scan.capacity) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    auto error = cudaMemsetAsync(view.hash, 255, std::size_t(view.scan.hash_capacity) * sizeof(*view.hash), stream);

    if (error) return int(error);
    prepare_resident_restore<<<1, 1, 0, stream>>>(view.scan.state, count);
    error = cudaGetLastError();

    if (error) return int(error);
    if (count) seed_resident_keys<<<(count + 127) / 128, 128, 0, stream>>>(view, count);
    error = cudaGetLastError();

    if (error) return int(error);
    publish_resident_keys<<<1, 1, 0, stream>>>(view.scan.state);
    return int(cudaGetLastError());
}
int cuda_sensor_reset_soa(resident_discovery_view view, storage::device_descriptor tiles, void* stream)
{
    if (!valid_resident_discovery(view) || tiles.record_schema != storage::schema::tsdf ||
        tiles.storage_layout != storage::layout::pool_soa || tiles.size_tiles != view.scan.capacity)
        return int(cudaErrorInvalidValue);
    auto error = cuda_sensor_restore_tiles(view, 0, stream);
    if (!error) error = storage::initialize_device_fields(tiles, stream);
    return error;
}
int cuda_sensor_allocate_tiles(resident_discovery_view view, void* opaque_stream)
{
    if (!valid_resident_discovery(view)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    auto error = detail::sort_tile_keys(view.scan.keys, view.scratch_keys, view.missing,
        view.offsets, view.sums, view.scan.capacity, stream);

    if (error) return int(error);
    const auto blocks = (view.scan.capacity + 127) / 128;
    classify_resident_keys<<<blocks, 128, 0, stream>>>(view);
    error = cudaGetLastError();
    if (!error) error = detail::scan_key_flags(view.missing, view.offsets, view.sums, view.scan.capacity, stream);

    if (error) return int(error);
    validate_resident_allocation<<<1, 1, 0, stream>>>(view);
    error = cudaGetLastError();

    if (error) return int(error);
    commit_resident_keys<<<blocks, 128, 0, stream>>>(view);
    error = cudaGetLastError();

    if (error) return int(error);
    publish_resident_keys<<<1, 1, 0, stream>>>(view.scan.state);
    return int(cudaGetLastError());
}
}
