#pragma once
#include "candidate_list.h"
#include "candidates.cuh"
#include <type_traits>

namespace covoxslam::sensor_grid
{
template<class real>
__device__ void list_bounds(field_view source, unsigned id, const candidate_transform& p,
    float guard, int (&lower)[3], int (&upper)[3])
{
    const real coordinate[]{real(p.origin) + real(id % source.side) + real(.5),
        real(p.origin) + real(id / source.side % source.side) + real(.5),
        real(p.origin) + real(id / (source.side * source.side)) + real(.5)};
    bool ambiguous{};

    for (unsigned i = 0; i < 3; ++i) {
        const real center = real(p.translation[i]) + real(p.matrix[i][0]) * coordinate[0] +
            real(p.matrix[i][1]) * coordinate[1] + real(p.matrix[i][2]) * coordinate[2];
        const real lo = center - real(p.extent[i]) - real(.5), hi = center + real(p.extent[i]) - real(.5);

        if constexpr (std::is_same_v<real, float>) {
            ambiguous |= !isfinite(lo) || !isfinite(hi) ||
                fabsf(lo - nearbyintf(lo)) <= guard || fabsf(hi - nearbyintf(hi)) <= guard;
        }
        lower[i] = int(ceil(lo)); upper[i] = int(floor(hi));
    }

    if constexpr (std::is_same_v<real, float>)
        if (ambiguous) list_bounds<double>(source, id, p, guard, lower, upper);
}
__device__ inline void list_emit(unsigned word, unsigned bits, candidate_list_view list)
{
    const unsigned count = __popc(bits);

    if (!count) return;
    const auto first = atomicAdd(&list.counters->count, static_cast<unsigned long long>(count));

    if (first + count > list.capacity) atomicExch(&list.counters->overflow, 1U);
    unsigned offset{};

    while (bits) {
        const unsigned bit = unsigned(__ffs(bits) - 1);

        if (first + offset < list.capacity) list.ids[first + offset] = word * 32 + bit;
        bits &= bits - 1;
        ++offset;
    }
}
__device__ inline void grouped_mask_or(unsigned word, unsigned bits, std::uint32_t* masks)
{
#if __CUDA_ARCH__ >= 700
    const unsigned peers = __match_any_sync(__activemask(), word);
    unsigned merged{};

    for (unsigned remaining = peers; remaining; remaining &= remaining - 1)
        merged |= __shfl_sync(peers, bits, __ffs(remaining) - 1);

    if ((threadIdx.x & 31U) != unsigned(__ffs(peers) - 1)) return;
    bits = merged;
#endif
    atomicOr(masks + word, bits);
}
template<class real>
__global__ void list_mark(field_view source, const std::uint32_t* active, std::uint32_t active_count,
    candidate_transform transform, float guard, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, candidate_list_view list)
{
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= active_count) return;
    const auto id = active[index];
    const auto value = source.voxels[id];

    if (!value.contributions || !(value.weight > 0) || !isfinite(value.weight) || !isfinite(value.distance)) return;
    int lo[3]{}, hi[3]{};
    list_bounds<real>(source, id, transform, guard, lo, hi);

    for (int z = candidate_divide(lo[2]); z <= candidate_divide(hi[2]); ++z)
        for (int y = candidate_divide(lo[1]); y <= candidate_divide(hi[1]); ++y)
            for (int x = candidate_divide(lo[0]); x <= candidate_divide(hi[0]); ++x) {
                const auto slot = candidate_slot({x, y, z}, slots, count, keys);

                if (slot < 0) continue;
                const int lx = max(0, lo[0] - x*8), ux = min(7, hi[0] - x*8);
                const int ly = max(0, lo[1] - y*8), uy = min(7, hi[1] - y*8);
                const int lz = max(0, lo[2] - z*8), uz = min(7, hi[2] - z*8);
                const unsigned x_bits = ((1U << (ux - lx + 1)) - 1U) << lx;

                for (int tz = lz; tz <= uz; ++tz)
                    for (int half = ly / 4; half <= uy / 4; ++half) {
                        unsigned bits{};

                        for (int ty = max(ly, half * 4); ty <= min(uy, half * 4 + 3); ++ty)
                            bits |= x_bits << (8 * (ty % 4));
                        grouped_mask_or(unsigned(slot) * 16 + unsigned(tz) * 2 + unsigned(half), bits, list.masks);
                    }
            }
}
__global__ void compact_candidate_bits(const std::uint32_t* slots, candidate_list_view list)
{
    if (threadIdx.x >= 16) return;
    const auto word = slots[blockIdx.x] * 16 + threadIdx.x;
    list_emit(word, list.masks[word], list);
}
inline float list_boundary_guard(field_view source, candidate_transform p)
{
    const double source_scale = std::max(1.0, std::max(std::abs(p.origin), std::abs(p.origin + source.side)) + 2);
    double scale = source_scale;

    for (unsigned i = 0; i < 3; ++i)
        scale = std::max(scale, std::abs(p.translation[i]) + source_scale *
            (std::abs(p.matrix[i][0]) + std::abs(p.matrix[i][1]) + std::abs(p.matrix[i][2])) + 2);
    return float(64 * FLT_EPSILON * scale + 16 * FLT_EPSILON);
}
}
