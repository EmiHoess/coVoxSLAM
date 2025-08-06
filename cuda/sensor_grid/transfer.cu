#include "map_native.h"
#include "resident.h"
#include "tsdf_views.cuh"
#include "../common/geometry.cuh"

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
__global__ void transfer_cells(field_view source, const std::uint32_t* slots,
    const detail::cuda_tile_coordinate* keys, sensor_tsdf_view tiles,
    detail::cuda_projective_pose transform, float size, float maximum_weight, float band,
    unsigned long long* count, const discovery_state* resident_state)
{
    if (resident_state->error || blockIdx.x / 4 >= resident_state->count) return;
    const auto slot = slots[blockIdx.x / 4], local = (blockIdx.x % 4) * 128 + threadIdx.x;
    const auto key = keys[slot];
    const float3 point{(key.x * 8 + int(local % 8) + .5f) * size,
        (key.y * 8 + int(local / 8 % 8) + .5f) * size, (key.z * 8 + int(local / 64) + .5f) * size};
    const auto sensor = detail::projective_reference_point(transform, point);
    const auto measured = sample(source, sensor.x, sensor.y, sensor.z, interpolation::nearest);
    bool updated{};

    if (measured.valid)
    {
        auto value = storage::gather_record(tiles[slot].voxels[local]);
        const float old = value.flags & 1U ? value.weight : 0.f;
        const double total = double(old) + measured.weight;
        value.distance = static_cast<float>((double(old) * (old > 0 ? value.distance : 0.f) +
            double(measured.weight) * measured.distance) / total);
        value.weight = static_cast<float>(fmin(double(maximum_weight), total));

        value.flags = 1U | (fabsf(value.distance) <= band ? 2U : 0U);
        if (value.sample_count != UINT32_MAX) ++value.sample_count;
         storage::scatter_record(tiles[slot].voxels[local], value);
        updated = true;
    }
    const unsigned updates = __syncthreads_count(updated);
    if (!threadIdx.x && updates) atomicAdd(count, static_cast<unsigned long long>(updates));
}
}
int cuda_sensor_resident_soa(field_view source, resident_discovery_view view,
    storage::device_descriptor tiles, detail::cuda_projective_pose transform, float size,
    float maximum_weight, float band, unsigned long long* updates, void* stream)
{
    if (!valid_resident_discovery(view) || !sampling_detail::valid_view(source) ||
        tiles.record_schema != storage::schema::tsdf || tiles.storage_layout != storage::layout::pool_soa ||
        tiles.size_tiles < view.scan.capacity || tiles.size_tiles > tiles.capacity_tiles || !tiles.generation ||
        tiles.tile_stride != 2048 || tiles.voxel_stride != 4 || !updates ||
        !(size > 0) || !isfinite(size) || !(maximum_weight > 0) || !isfinite(maximum_weight) ||
        !(band >= 0) || !isfinite(band)) return int(cudaErrorInvalidValue);

    for (unsigned i = 0; i < 4; ++i)
        if (!tiles.fields[i]) return int(cudaErrorInvalidValue);
    sensor_tsdf_view target{storage::device_accessor<storage::schema::tsdf>{tiles}};
    transfer_cells<<<view.scan.capacity * 4, 128, 0, static_cast<cudaStream_t>(stream)>>>(
        source, view.slots, view.keys, target, transform, size, maximum_weight, band, updates, view.scan.state);
    return int(cudaGetLastError());
}
int cuda_sensor_mask(field_view source, std::uint8_t* flags, void* stream)
{
    if (!source.voxels || !source.side || source.side > 1624 || !flags) return int(cudaErrorInvalidValue);
    const auto n = (source.side + 7) / 8;
    mask_cells<<<n * n * n, 512, 0, static_cast<cudaStream_t>(stream)>>>(source, n, flags);
    return int(cudaGetLastError());
}
}
