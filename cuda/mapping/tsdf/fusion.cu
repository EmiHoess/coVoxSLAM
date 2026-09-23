#include "fusion.cuh"
#include "allocation.cuh"
#include <covoxslam/storage/views.hpp>
#include <covoxslam/voxel/flags.hpp>
#include <covoxslam/voxel/tile_shape.hpp>
#include "detail/geometry.cuh"

namespace covoxslam::sensor_grid
{
using sensor_tsdf_view = storage::device_map_view<storage::schema::tsdf>;
namespace
{
inline constexpr unsigned tile_kernel_threads = 128;
inline constexpr unsigned tile_kernel_blocks = unsigned(tile_voxels) / tile_kernel_threads;
static_assert(tile_voxels % tile_kernel_threads == 0);
static_assert(tile_voxels <= 1024);

__global__ void mask_cells(field_view source, unsigned n, std::uint8_t* flags)
{
    const auto block = blockIdx.x, local = threadIdx.x;
    const auto x = block % n * tile_edge + local % tile_edge;
    const auto y = block / n % n * tile_edge + local / tile_edge % tile_edge;
    const auto z = block / (n * n) * tile_edge + local / tile_face_voxels;
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
    if (resident_state->error || blockIdx.x / tile_kernel_blocks >= resident_state->count) return;
    const auto slot = slots[blockIdx.x / tile_kernel_blocks];
    const auto local = (blockIdx.x % tile_kernel_blocks) * tile_kernel_threads + threadIdx.x;
    const auto key = keys[slot];
    const float3 point{(key.x * tile_edge + int(local % tile_edge) + .5f) * size,
        (key.y * tile_edge + int(local / tile_edge % tile_edge) + .5f) * size,
        (key.z * tile_edge + int(local / tile_face_voxels) + .5f) * size};
    const auto sensor = detail::projective_reference_point(transform, point);
    const auto measured = sample(source, sensor.x, sensor.y, sensor.z, interpolation::nearest);
    bool updated{};

    if (measured.valid)
    {
        auto value = storage::gather_record(tiles[slot].voxels[local]);
        const float old = value.flags & observed_flag ? value.weight : 0.f;
        const double total = double(old) + measured.weight;
        value.distance = static_cast<float>((double(old) * (old > 0 ? value.distance : 0.f) +
            double(measured.weight) * measured.distance) / total);
        value.weight = static_cast<float>(fmin(double(maximum_weight), total));

        value.flags = observed_flag | (fabsf(value.distance) <= band ? near_surface_flag : 0U);
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
    float maximum_weight, float band, unsigned long long* updates,
    std::uint64_t& kernel_launches, void* stream)
{
    if (!valid_resident_discovery(view) || !sampling_detail::valid_view(source) ||
        tiles.record_schema != storage::schema::tsdf ||
        !storage::valid_pool_soa_descriptor_shape(tiles) ||
        tiles.size_tiles < view.scan.capacity || !tiles.generation || !updates ||
        !(size > 0) || !isfinite(size) || !(maximum_weight > 0) || !isfinite(maximum_weight) ||
        !(band >= 0) || !isfinite(band)) return int(cudaErrorInvalidValue);

    for (unsigned i = 0; i < storage::field_count<storage::schema::tsdf>; ++i)
        if (!tiles.fields[i]) return int(cudaErrorInvalidValue);
    sensor_tsdf_view target{storage::device_accessor<storage::schema::tsdf>{tiles}};
    transfer_cells<<<view.scan.capacity * tile_kernel_blocks, tile_kernel_threads,
        0, static_cast<cudaStream_t>(stream)>>>(
        source, view.slots, view.keys, target, transform, size, maximum_weight, band, updates, view.scan.state);
    const auto error = cudaGetLastError();
    if (!error) ++kernel_launches;
    return int(error);
}
int cuda_sensor_mask(
    field_view source, std::uint8_t* flags, std::uint64_t& kernel_launches, void* stream)
{
    if (!sampling_detail::valid_view(source) || !flags) return int(cudaErrorInvalidValue);
    const auto n = (source.side + tile_edge - 1) / tile_edge;
    mask_cells<<<n * n * n, unsigned(tile_voxels), 0,
        static_cast<cudaStream_t>(stream)>>>(source, n, flags);
    const auto error = cudaGetLastError();
    if (!error) ++kernel_launches;
    return int(error);
}
}
