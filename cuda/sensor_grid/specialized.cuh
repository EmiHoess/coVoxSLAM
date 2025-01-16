#pragma once
#include "../projective/geometry.cuh"
#include <covoxslam/sensor_grid_sampling.h>

namespace covoxslam::sensor_grid
{
struct affine_transfer { float3 axes[3]; };
inline affine_transfer prepare_affine(detail::cuda_projective_pose transform, float size)
{
    transform.translation[0] = transform.translation[1] = transform.translation[2] = 0;
    return {{detail::projective_reference_point(transform, {size, 0, 0}),
        detail::projective_reference_point(transform, {0, size, 0}),
        detail::projective_reference_point(transform, {0, 0, size})}};
}
__device__ inline float coordinate_error_bound(field_view source,
    detail::cuda_projective_pose transform, float3 point, float size)
{
    const float world_scale = fmaxf(fabsf(point.x), fmaxf(fabsf(point.y), fabsf(point.z)));
    const float pose_scale = fmaxf(fabsf(transform.translation[0]),
        fmaxf(fabsf(transform.translation[1]), fabsf(transform.translation[2])));
    return 64 * FLT_EPSILON * fmaxf(1.f,
        fmaxf((world_scale + pose_scale + 16 * size) / source.voxel_size,
            fabsf(source.origin / source.voxel_size)));
}
__device__ inline sample_value specialized_sample(field_view view, float3 point, float error, bool& reference)
{
    using namespace sampling_detail;
    const float u[]{(point.x - view.origin) / view.voxel_size - .5f,
        (point.y - view.origin) / view.voxel_size - .5f, (point.z - view.origin) / view.voxel_size - .5f};
    int base[3]{};
    float fraction[3]{};

    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (!finite(u[axis]) || !finite(error) || fabsf(u[axis] - floorf(u[axis] + .5f)) <= error + .001f) {
            reference = true;
            return {};
        }

        if (u[axis] < -1 || !(u[axis] < float(view.side))) return {};
        const float lower = floorf(u[axis]);
        base[axis] = int(lower); fraction[axis] = u[axis] - lower;
    }
    accumulator result;

    for (unsigned corner = 0; corner < 8; ++corner) {
        const unsigned dx = corner & 1, dy = corner >> 1 & 1, dz = corner >> 2;
        const float alpha = (dx ? fraction[0] : 1 - fraction[0]) *
            (dy ? fraction[1] : 1 - fraction[1]) * (dz ? fraction[2] : 1 - fraction[2]);

        if (fabsf(alpha - 8 * FLT_EPSILON) <= 4 * error) { reference = true; return {}; }

        if (!result.add(view, base[0] + int(dx), base[1] + int(dy), base[2] + int(dz), alpha)) return {};
    }
    return result.finish();
}
__global__ __launch_bounds__(128) void specialized_transfer(field_view source, const std::uint32_t* slots,
    const detail::cuda_tile_coordinate* keys, detail::cuda_tsdf_tile* tiles,
    detail::cuda_projective_pose transform, affine_transfer affine, float size,
    float maximum_weight, float band, unsigned long long* count)
{
    __shared__ unsigned slot;
    __shared__ detail::cuda_tile_coordinate key;
    __shared__ float3 base;
    __shared__ float error;

    if (!threadIdx.x) {
        slot = slots[blockIdx.x]; key = keys[slot];
        const float3 point{(key.x * 8 + .5f) * size, (key.y * 8 + .5f) * size, (key.z * 8 + .5f) * size};
        base = detail::projective_reference_point(transform, point);
        error = coordinate_error_bound(source, transform, point, size);
    }
    __syncthreads();
    unsigned updated{};

    for (unsigned local = threadIdx.x; local < 512; local += 128)
    {
        const float x = float(local % 8), y = float(local / 8 % 8), z = float(local / 64);
        const float3 point{
            ((base.x + x * affine.axes[0].x) + y * affine.axes[1].x) + z * affine.axes[2].x,
            ((base.y + x * affine.axes[0].y) + y * affine.axes[1].y) + z * affine.axes[2].y,
            ((base.z + x * affine.axes[0].z) + y * affine.axes[1].z) + z * affine.axes[2].z};
        bool reference{};
        auto measured = specialized_sample(source, point, error, reference);

        if (reference) {
            const float3 original{(key.x * 8 + int(local % 8) + .5f) * size,
                (key.y * 8 + int(local / 8 % 8) + .5f) * size, (key.z * 8 + int(local / 64) + .5f) * size};
            const auto sensor = detail::projective_reference_point(transform, original);
            measured = sample(source, sensor.x, sensor.y, sensor.z, interpolation::trilinear);
        }

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
        ++updated;
    }

    for (unsigned delta = 16; delta; delta /= 2) updated += __shfl_down_sync(0xffffffffU, updated, delta);

    if (!(threadIdx.x & 31U) && updated) atomicAdd(count, static_cast<unsigned long long>(updated));
}
}
