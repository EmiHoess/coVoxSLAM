#include "discovery.h"
#include "../ray_resident/hash.cuh"
#include <cmath>
#include <climits>
#include <math_constants.h>

namespace covoxslam::sensor_grid
{
namespace
{
__device__ float3 discovery_cross(float3 a, float3 b)
{
    return {__fsub_rn(__fmul_rn(a.y, b.z), __fmul_rn(a.z, b.y)),
        __fsub_rn(__fmul_rn(a.z, b.x), __fmul_rn(a.x, b.z)),
        __fsub_rn(__fmul_rn(a.x, b.y), __fmul_rn(a.y, b.x))};
}
__device__ float3 discovery_point(detail::cuda_projective_pose pose, float3 point)
{
    const float3 q{pose.rotation_xyzw[0], pose.rotation_xyzw[1], pose.rotation_xyzw[2]};
    auto twice = discovery_cross(q, point);
    twice = {__fmul_rn(twice.x, 2), __fmul_rn(twice.y, 2), __fmul_rn(twice.z, 2)};
    const auto cross = discovery_cross(q, twice);
    const auto w = pose.rotation_xyzw[3];
    return {__fadd_rn(pose.translation[0], __fadd_rn(__fadd_rn(point.x, __fmul_rn(twice.x, w)), cross.x)),
        __fadd_rn(pose.translation[1], __fadd_rn(__fadd_rn(point.y, __fmul_rn(twice.y, w)), cross.y)),
        __fadd_rn(pose.translation[2], __fadd_rn(__fadd_rn(point.z, __fmul_rn(twice.z, w)), cross.z))};
}
__device__ void discover_key(discovery_view view, detail::cuda_tile_coordinate coordinate)
{
    const auto key = detail::resident_key({coordinate.z, coordinate.y, coordinate.x});

    if (key == UINT64_MAX) { atomicOr(&view.state->error, discovery_coordinate); return; }
    auto bucket = std::uint32_t(detail::resident_hash(key) & (view.hash_capacity - 1));

    for (std::uint32_t probe = 0; probe < view.hash_capacity; ++probe)
    {
        const auto old = atomicCAS(reinterpret_cast<unsigned long long*>(view.hash + bucket), UINT64_MAX, key);

        if (old == key) return;
        if (old == UINT64_MAX) {
            const auto position = atomicAdd(&view.state->count, 1U);

            if (position < view.capacity) view.keys[position] = key;
            else atomicOr(&view.state->error, discovery_capacity);
            return;
        }
        bucket = (bucket + 1) & (view.hash_capacity - 1);
    }
    atomicOr(&view.state->error, discovery_hash);
}
__global__ void discover_sensor_tiles(field_view source, const std::uint8_t* observed,
    unsigned n, detail::cuda_projective_pose pose, discovery_view view)
{
    const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index >= std::size_t(n) * n * n || !observed[index]) return;
    const unsigned cell[]{unsigned(index % n) * 8, unsigned(index / n % n) * 8, unsigned(index / (n * n)) * 8};
    float lo[]{CUDART_INF_F, CUDART_INF_F, CUDART_INF_F}, hi[]{-CUDART_INF_F, -CUDART_INF_F, -CUDART_INF_F};

    for (unsigned corner = 0; corner < 8; ++corner) {
        float p[3]{};

        for (unsigned a = 0; a < 3; ++a)
            p[a] = __fadd_rn(source.origin, __fmul_rn(__fadd_rn(float(cell[a]),
                corner & (1U << a) ? 9.f : -1.f), source.voxel_size));
        const auto v = discovery_point(pose, {p[0], p[1], p[2]});
        const float values[]{v.x, v.y, v.z};

        for (unsigned a = 0; a < 3; ++a) {

            if (!isfinite(values[a])) { atomicOr(&view.state->error, discovery_coordinate); return; }
            lo[a] = fminf(lo[a], values[a]); hi[a] = fmaxf(hi[a], values[a]);
        }
    }
    int lower[3]{}, upper[3]{};
    std::size_t volume{1};

    for (unsigned a = 0; a < 3; ++a)
    {
        const double first = floor(__ddiv_rn(lo[a], double(source.voxel_size) * 8));
        const double last = floor(__ddiv_rn(hi[a], double(source.voxel_size) * 8));

        if (!isfinite(first) || !isfinite(last) || first < -1048576 || last >= 1048576 || last < first) {
            atomicOr(&view.state->error, discovery_coordinate); return;
        }
        lower[a] = int(first); upper[a] = int(last);
        const auto extent = std::size_t(upper[a] - lower[a]) + 1;

        if (extent > view.capacity / volume) { atomicOr(&view.state->error, discovery_capacity); return; }
        volume *= extent;
    }

    for (int z = lower[2]; z <= upper[2]; ++z)
        for (int y = lower[1]; y <= upper[1]; ++y)
            for (int x = lower[0]; x <= upper[0]; ++x) {

                if (atomicAdd(&view.state->error, 0U)) return;
                discover_key(view, {x, y, z});
            }
}
}
int cuda_sensor_discover(field_view source, const std::uint8_t* observed,
    detail::cuda_projective_pose transform, discovery_view view, void* opaque_stream)
{
    if (!sampling_detail::valid_view(source) || source.side > 1624 || !observed || !view.hash || !view.keys ||
        !view.state || !view.capacity || view.capacity > 1048576 || view.hash_capacity != discovery_hash_size(view.capacity))
        return int(cudaErrorInvalidValue);
    const auto q = transform.rotation_xyzw;
    const double norm = std::hypot(std::hypot(double(q[0]), double(q[1])), std::hypot(double(q[2]), double(q[3])));

    if (!std::isfinite(norm) || std::abs(norm - 1) > .001) return int(cudaErrorInvalidValue);

    for (auto& value : transform.rotation_xyzw) value = float(value / norm);

    for (float value : transform.translation)
        if (!std::isfinite(value)) return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(opaque_stream);
    auto error = cudaMemsetAsync(view.state, 0, 2 * sizeof(std::uint32_t), stream);
    if (!error) error = cudaMemsetAsync(view.hash, 255, std::size_t(view.hash_capacity) * 8, stream);
    if (!error) error = cudaMemsetAsync(view.keys, 255, std::size_t(view.capacity) * 8, stream);

    if (error) return int(error);
    const unsigned n = (source.side + 7) / 8;
    discover_sensor_tiles<<<(n * n * n + 127) / 128, 128, 0, stream>>>(source, observed, n, transform, view);
    return int(cudaGetLastError());
}
}
