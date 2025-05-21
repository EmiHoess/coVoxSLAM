#pragma once
#include "band_esdf_views.cuh"
#include "../esdf_site/coordinates.cuh"

namespace covoxslam::store
{
struct band_tile_slot
{
    unsigned source{}, target{};
};
__global__ void index_band_tiles(const block_key* keys, unsigned count,
    device_index_view input, device_index_view output, std::size_t input_size,
    std::size_t output_size, detail::cuda_tile_coordinate* coordinates,
    band_tile_slot* slots, unsigned* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= count) return;
    const auto key = keys[i];
    const auto source = find_block(input, key);
     const auto target = find_block(output, key);
    const detail::cuda_tile_coordinate coordinate{key.x, key.y, key.z};
    bool valid = source.found && target.found && source.reuse_generation && target.reuse_generation &&
        source.slot < input_size && target.slot < output_size;

    if (i) {
        const auto previous = keys[i - 1];
        valid = valid && detail::site_tile_less({previous.x, previous.y, previous.z}, coordinate);
    }
    if (!valid)
    {
        atomicOr(status, 1U);
        return;
    }
    coordinates[i] = coordinate;
    slots[i] = {source.slot, target.slot};
}
}
