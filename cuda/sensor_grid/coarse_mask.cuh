#pragma once
#include "candidates.cuh"

namespace covoxslam::sensor_grid
{
__global__ void reset_coarse_candidates(const std::uint32_t* slots, std::uint32_t* masks, bool full)
{
    if (!threadIdx.x) masks[slots[blockIdx.x]] = full ? 255U : 0U;
}
__global__ void mark_coarse_candidates(field_view source, const std::uint32_t* active,
    std::uint32_t active_count, candidate_transform transform, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, std::uint32_t* masks)
{
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= active_count) return;
    const auto id = active[index];
    const auto value = source.voxels[id];

    if (!value.contributions || !(value.weight > 0) || !isfinite(value.weight) || !isfinite(value.distance)) return;
    const double coordinate[]{transform.origin + id % source.side + .5,
        transform.origin + id / source.side % source.side + .5,
        transform.origin + id / (source.side * source.side) + .5};
    int lower[3]{}, upper[3]{};

    for (unsigned i = 0; i < 3; ++i) {
        const double center = transform.translation[i] + transform.matrix[i][0] * coordinate[0] +
            transform.matrix[i][1] * coordinate[1] + transform.matrix[i][2] * coordinate[2];
        lower[i] = int(ceil(center - transform.extent[i] - .5));
         upper[i] = int(floor(center + transform.extent[i] - .5));

        if (lower[i] > upper[i]) return;
    }

    for (int z = candidate_divide(lower[2]); z <= candidate_divide(upper[2]); ++z)
        for (int y = candidate_divide(lower[1]); y <= candidate_divide(upper[1]); ++y)
            for (int x = candidate_divide(lower[0]); x <= candidate_divide(upper[0]); ++x)
            {
                const auto slot = candidate_slot({x, y, z}, slots, count, keys);

                if (slot < 0) continue;
                const int lo[]{max(0, lower[0] - x * 8) / 4, max(0, lower[1] - y * 8) / 4,
                    max(0, lower[2] - z * 8) / 4};
                const int hi[]{min(7, upper[0] - x * 8) / 4, min(7, upper[1] - y * 8) / 4,
                    min(7, upper[2] - z * 8) / 4};
                unsigned bits{};

                for (int k = lo[2]; k <= hi[2]; ++k)
                    for (int j = lo[1]; j <= hi[1]; ++j)
                        for (int i = lo[0]; i <= hi[0]; ++i) bits |= 1U << (i + j * 2 + k * 4);

                if (bits) atomicOr(masks + slot, bits);
            }
}
}
