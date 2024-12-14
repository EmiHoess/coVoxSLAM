#include "map_native.h"
#include <covoxslam/sensor_grid_overlap.h>
#include "../projective/geometry.cuh"

namespace covoxslam::sensor_grid
{
namespace
{
__global__ void mask_cells(field_view source, unsigned n, std::uint8_t* flags)
{
    const auto block = blockIdx.x, local = threadIdx.x;
    const auto x = block % n * 8 + local % 8;
    const auto y = block / n % n * 8 + local / 8 % 8;
    const auto z = block / (n * n) * 8 + local / 64;
    bool observed{};

    if (x < source.side && y < source.side && z < source.side) {
        const auto value = source.voxels[x + std::size_t(source.side) * (y + std::size_t(source.side) * z)];
        observed = value.contributions && value.weight > 0 && isfinite(value.weight) && isfinite(value.distance);
    }
    const auto any = __syncthreads_or(observed);
    if (!local) flags[block] = any ? 1 : 0;
}
__global__ void transfer_cells(field_view source, interpolation mode, const std::uint32_t* slots,
    const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    unsigned long long* count)
{
    const auto slot = slots[blockIdx.x], local = threadIdx.x;
    const auto key = keys[slot];
    const float3 point{(key.x * 8 + int(local % 8) + .5f) * size,
        (key.y * 8 + int(local / 8 % 8) + .5f) * size, (key.z * 8 + int(local / 64) + .5f) * size};
    const auto sensor = detail::projective_reference_point(transform, point);
    auto measured = sample(source, sensor.x, sensor.y, sensor.z, mode);
    if (mode == interpolation::exact_overlap)
    {
        overlap_box box{{sensor.x, sensor.y, sensor.z}, {}, size};
        auto rotation = transform;
        rotation.translation[0] = rotation.translation[1] = rotation.translation[2] = 0;

        for (unsigned axis = 0; axis < 3; ++axis) {
            const auto v = detail::projective_reference_point(rotation, {axis == 0 ? 1.f : 0.f,
                axis == 1 ? 1.f : 0.f, axis == 2 ? 1.f : 0.f});
            box.axes[axis][0] = v.x; box.axes[axis][1] = v.y; box.axes[axis][2] = v.z;
        }
        measured = sample_overlap(source, box);
    }
    bool updated{};

    if (measured.valid)
    {
        auto value = tiles[slot].voxels[local];
        const float old = value.flags & 1U ? value.weight : 0.f;
        const double total = double(old) + measured.weight;
        value.distance = static_cast<float>((double(old) * (old > 0 ? value.distance : 0.f) +
            double(measured.weight) * measured.distance) / total);
        value.weight = static_cast<float>(fmin(double(maximum_weight), total));
        value.flags = 1U | (fabsf(value.distance) <= band ? 2U : 0U);
        if (value.sample_count != UINT32_MAX) ++value.sample_count;
         tiles[slot].voxels[local] = value;
        updated = true;
    }
    const unsigned updates = __syncthreads_count(updated);
    if (!local && updates) atomicAdd(count, static_cast<unsigned long long>(updates));
}
}
int cuda_sensor_mask(field_view source, std::uint8_t* flags, void* stream)
{
    if (!source.voxels || !source.side || source.side > 1624 || !flags) return int(cudaErrorInvalidValue);
    const auto n = (source.side + 7) / 8;
    mask_cells<<<n * n * n, 512, 0, static_cast<cudaStream_t>(stream)>>>(source, n, flags);
    return int(cudaGetLastError());
}
int cuda_sensor_transfer(field_view source, interpolation mode, const std::uint32_t* slots,
    std::uint32_t count, const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    unsigned long long* updates, void* stream)
{
    if (!count) return int(cudaSuccess);
    if (!source.voxels || !slots || !keys || !tiles || !updates || unsigned(mode) > 3 ||
        !(size > 0) || !isfinite(size) || !(maximum_weight > 0) || !isfinite(maximum_weight) ||
        !(band >= 0) || !isfinite(band)) return int(cudaErrorInvalidValue);
    transfer_cells<<<count, 512, 0, static_cast<cudaStream_t>(stream)>>>(source, mode, slots, keys, tiles,
        transform, size, maximum_weight, band, updates);
    return int(cudaGetLastError());
}
}
