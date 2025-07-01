#pragma once
#include "candidate_bounds.cuh"
#include "../common/geometry.cuh"

namespace covoxslam::sensor_grid
{
__device__ inline bool candidate_key_less(detail::cuda_tile_coordinate a, detail::cuda_tile_coordinate b)
{
    return a.z != b.z ? a.z < b.z : a.y != b.y ? a.y < b.y : a.x < b.x;
}
__device__ inline int candidate_slot(detail::cuda_tile_coordinate key, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys)
{
    std::uint32_t lo{}, hi = count;

    while (lo < hi) {
        const auto mid = lo + (hi - lo) / 2;

        if (candidate_key_less(keys[slots[mid]], key)) lo = mid + 1;
        else hi = mid;
    }

    if (lo == count) return -1;
    const auto found = keys[slots[lo]];
    return found.x == key.x && found.y == key.y && found.z == key.z ? int(slots[lo]) : -1;
}
__device__ inline int candidate_divide(int value) { return value / 8 - (value % 8 < 0); }
__global__ void reset_candidates(const std::uint32_t* slots, std::uint32_t* masks, bool full)
{
    if (threadIdx.x < 16) masks[std::size_t(slots[blockIdx.x]) * 16 + threadIdx.x] = full ? UINT32_MAX : 0;
}
__global__ void mark_candidates(field_view source, const std::uint32_t* active, std::uint32_t active_count,
    candidate_transform transform, const std::uint32_t* slots, std::uint32_t count,
    const detail::cuda_tile_coordinate* keys, std::uint32_t* masks)
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
    }

    for (int z = lower[2]; z <= upper[2]; ++z)
        for (int y = lower[1]; y <= upper[1]; ++y)
            for (int x = lower[0]; x <= upper[0]; ++x) {
                const detail::cuda_tile_coordinate key{candidate_divide(x), candidate_divide(y), candidate_divide(z)};
                const auto slot = candidate_slot(key, slots, count, keys);

                if (slot < 0) continue;
                const unsigned local = unsigned(x - key.x * 8) + unsigned(y - key.y * 8) * 8 + unsigned(z - key.z * 8) * 64;
                atomicOr(masks + std::size_t(slot) * 16 + local / 32, 1U << (local % 32));
            }
}
__global__ void compacted_candidate_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    const detail::cuda_tile_coordinate* keys, const std::uint32_t* masks, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    unsigned long long* updates, unsigned long long* candidates)
{
    if (threadIdx.x >= 16) return;
    const auto slot = slots[blockIdx.x], word = threadIdx.x;
    auto bits = masks[std::size_t(slot) * 16 + word];
    const auto key = keys[slot];
    unsigned changed{};
    atomicAdd(candidates, static_cast<unsigned long long>(__popc(bits)));

    while (bits) {
        const unsigned local = word * 32 + unsigned(__ffs(bits) - 1);
        bits &= bits - 1;
        const float3 point{(key.x * 8 + int(local % 8) + .5f) * size,
            (key.y * 8 + int(local / 8 % 8) + .5f) * size, (key.z * 8 + int(local / 64) + .5f) * size};
        const auto sensor = detail::projective_reference_point(transform, point);
        const auto measured = sample(source, sensor.x, sensor.y, sensor.z, mode);

        if (!measured.valid) continue;
        auto value = tiles[slot].voxels[local];
        const float old = value.flags & 1U ? value.weight : 0.f;
        const double total = double(old) + measured.weight;
        value.distance = float((double(old) * (old > 0 ? value.distance : 0.f) +
            double(measured.weight) * measured.distance) / total);
        value.weight = float(fmin(double(maximum_weight), total));
        value.flags = 1U | (fabsf(value.distance) <= band ? 2U : 0U);

        if (value.sample_count != UINT32_MAX) ++value.sample_count;
        tiles[slot].voxels[local] = value;
        ++changed;
    }

    if (changed) atomicAdd(updates, static_cast<unsigned long long>(changed));
}
}
