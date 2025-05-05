#include "mapping.h"
#include <cuda_runtime.h>
#include <climits>

namespace covoxslam::store
{
__device__ bool valid_esdf_key(device_index_view index, const block_key* keys, unsigned i, std::size_t size, unsigned& slot)
{
    const auto key = keys[i];
    const auto found = find_block(index, key);
    bool valid = found.found && found.slot < size;
    if (i) {
        const auto old = keys[i - 1];
        valid = valid && (old.z < key.z || (old.z == key.z && (old.y < key.y || (old.y == key.y && old.x < key.x))));
    }
    const int coordinates[]{key.x, key.y, key.z};

    for (auto coordinate : coordinates)
        valid = valid && std::int64_t(coordinate) * 8 >= INT_MIN && std::int64_t(coordinate) * 8 + 7 <= INT_MAX;
    slot = found.slot;
    return valid;
}
__global__ void pack_kernel(storage::device_map_view<storage::schema::tsdf, true> source,
    device_index_view index, const block_key* keys, unsigned count,
    detail::cuda_tile_coordinate* coordinates, detail::cuda_tsdf_tile* output, unsigned* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count) * 512) return;
    const auto tile = unsigned(i / 512), local = unsigned(i % 512);
    unsigned slot{};
    if (!valid_esdf_key(index, keys, tile, source.size(), slot)) { atomicOr(status, 1U); return; }
    if (!local) coordinates[tile] = {keys[tile].x, keys[tile].y, keys[tile].z};
    const auto value = source[slot].voxels[local];
    output[tile].voxels[local] = {value.distance, value.weight, value.sample_count, value.flags};
}
template<storage::schema s> __global__ void clear_esdf_kernel(storage::device_map_view<s> target)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < target.size() * 512) storage::scatter_record<s>(target[i / 512].voxels[i % 512], storage::value<s>{});
}
template<storage::schema s, class value_type> __global__ void unpack_kernel(storage::device_map_view<s> target,
    device_index_view index, const block_key* keys, unsigned count, const value_type* input, unsigned* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count) * 512) return;
    unsigned slot{};
    if (!valid_esdf_key(index, keys, unsigned(i / 512), target.size(), slot)) { atomicOr(status, 1U); return; }
    const auto value = input[i];
    if constexpr (s == storage::schema::blocks_esdf)
        storage::scatter_record<s>(target[slot].voxels[i % 512],
            {value.distance_m, value.parent[0], value.parent[1], value.parent[2], value.flags});
    else
        storage::scatter_record<s>(target[slot].voxels[i % 512],
            {value.squared_distance_voxels, value.parent[0], value.parent[1], value.parent[2], value.flags});
}
bool valid_esdf_binding(storage::device_descriptor payload, device_index_view index,
    const block_key* keys, unsigned count, unsigned* status, storage::schema expected)
{
    if (!keys || !count || count > UINT32_MAX / 512 || !status || !index.entries || !index.capacity ||
        (index.capacity & (index.capacity - 1)) || !index.allocation_generation || payload.record_schema != expected ||
        index.field != (expected == storage::schema::tsdf ? 0U : 1U) || payload.device != index.device ||
        payload.size_tiles > payload.capacity_tiles || payload.size_tiles > UINT32_MAX / 512 || !payload.generation) return false;
    int device{-1};
    if (cudaGetDevice(&device) != cudaSuccess || device != index.device) return false;
    const unsigned fields = expected == storage::schema::tsdf ? 4 : 5;

    for (unsigned f = 0; f < fields; ++f)
        if (!payload.fields[f] || reinterpret_cast<std::uintptr_t>(payload.fields[f]) % 4) return false;
    return payload.tile_stride && payload.tile_stride % 4 == 0 && payload.voxel_stride >= 4 && payload.voxel_stride % 4 == 0;
}
int pack_esdf_input(storage::device_descriptor payload, device_index_view index,
    const block_key* keys, std::uint32_t count, detail::cuda_tile_coordinate* ordered_keys,
    detail::cuda_tsdf_tile* values, unsigned* status, void* stream)
{
    if (!valid_esdf_binding(payload, index, keys, count, status, storage::schema::tsdf) || !ordered_keys || !values)
        return int(cudaErrorInvalidValue);
    const auto native = static_cast<cudaStream_t>(stream);
    auto error = cudaMemsetAsync(status, 0, sizeof(unsigned), native);
    if (error) return int(error);
    pack_kernel<<<count * 4, 128, 0, native>>>(storage::device_map_view<storage::schema::tsdf, true>{
        storage::device_accessor<storage::schema::tsdf, true>{payload}}, index, keys, count, ordered_keys, values, status);
    return int(cudaGetLastError());
}
template<storage::schema s, class value_type> int unpack_dispatch(storage::device_descriptor payload, device_index_view index,
    const block_key* keys, unsigned count, const value_type* values, unsigned* status, cudaStream_t stream)
{
    const storage::device_map_view<s> view{storage::device_accessor<s>{payload}};
    clear_esdf_kernel<s><<<unsigned(payload.size_tiles * 4), 128, 0, stream>>>(view);
    auto error = cudaGetLastError();
    if (error) return int(error);
    unpack_kernel<s><<<count * 4, 128, 0, stream>>>(view, index, keys, count, values, status);
    return int(cudaGetLastError());
}
int unpack_esdf_output(storage::device_descriptor payload, device_index_view index,
    const block_key* keys, std::uint32_t count, const void* values,
    storage::schema schema, unsigned* status, void* stream)
{
    if (!valid_esdf_binding(payload, index, keys, count, status, schema) || !values || !payload.size_tiles ||
        (schema != storage::schema::blocks_esdf && schema != storage::schema::projective_esdf)) return int(cudaErrorInvalidValue);
    const auto native = static_cast<cudaStream_t>(stream);
    auto error = cudaMemsetAsync(status, 0, sizeof(unsigned), native);
    if (error) return int(error);
    if (schema == storage::schema::blocks_esdf)
        return unpack_dispatch<storage::schema::blocks_esdf>(payload, index, keys, count,
            static_cast<const detail::cuda_blocks_voxel*>(values), status, native);
    return unpack_dispatch<storage::schema::projective_esdf>(payload, index, keys, count,
        static_cast<const detail::cuda_site_esdf_voxel*>(values), status, native);
}
}
