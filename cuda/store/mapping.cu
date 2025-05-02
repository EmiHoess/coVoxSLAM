#include "mapping.h"
#include "../ray_fusion/serial/fuse.cuh"
#include "../projective/geometry.cuh"

namespace covoxslam::store
{
using tsdf_view = storage::device_map_view<storage::schema::tsdf>;
__device__ std::uint32_t lookup_voxel(device_index_view index, const int* coordinates, unsigned& local)
{
    int block[3];
    local = 0;

    for (unsigned axis = 0, stride = 1; axis < 3; ++axis, stride *= 8) {
        auto offset = coordinates[axis] % 8;
        if (offset < 0) offset += 8;
        block[axis] = (coordinates[axis] - offset) / 8;
        local += unsigned(offset) * stride;
    }
    const auto found = find_block(index, {index.submap, block[0], block[1], block[2]});
    return found.found ? found.slot : UINT32_MAX;
}
template<bool validate> __global__ void ordered_kernel(tsdf_view field, device_index_view index,
    const detail::cuda_ray_contribution* samples, std::size_t sample_count,
    const detail::cuda_voxel_segment* segments, unsigned count, detail::cuda_fusion_parameters config, unsigned* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if constexpr (!validate) { if (*status) return; }
    const auto segment = segments[i];
    unsigned local{};
    const auto slot = lookup_voxel(index, segment.voxel, local);
    bool valid = slot < field.size() && segment.count && segment.begin <= sample_count && segment.count <= sample_count - segment.begin;
    if constexpr (validate) {
        if (!i) valid = valid && segment.begin == 0;
        else {
            const auto previous = segments[i - 1];
            valid = valid && previous.begin <= sample_count && previous.count <= sample_count - previous.begin &&
                segment.begin == previous.begin + previous.count && detail::voxel_key_less(previous, segment);
        }
        if (i + 1 == count) valid = valid && segment.begin + segment.count == sample_count;
    }
    if (!valid) { atomicOr(status, 1U); return; }
    if constexpr (validate) {

        for (unsigned j = 0; j < segment.count; ++j) {
            const auto sample = samples[segment.begin + j];

            for (unsigned axis = 0; axis < 3; ++axis) valid = valid && sample.voxel[axis] == segment.voxel[axis];
            if (j) {
                const auto prior = samples[segment.begin + j - 1];
                valid = valid && (prior.ray_ordinal < sample.ray_ordinal ||
                    (prior.ray_ordinal == sample.ray_ordinal && prior.traversal_ordinal <= sample.traversal_ordinal));
            }
        }
        if (!valid) { atomicOr(status, 1U); return; }
    }
    const auto target = field[slot].voxels[local];
    detail::cuda_tsdf_voxel value{target.distance, target.weight, target.sample_count, target.flags};
    if (!detail::fuse_ordered_samples(value, samples + segment.begin, segment.count, config)) { atomicOr(status, 2U); return; }
    if constexpr (!validate)
        storage::scatter_record<storage::schema::tsdf>(target, {value.distance, value.weight, value.sample_count, value.flags});
}
__global__ void validate_keys(device_index_view index, const block_key* keys, unsigned count, std::size_t size, unsigned* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto key = keys[i];
    const auto found = find_block(index, key);
    bool valid = key.submap == index.submap && found.found && found.slot < size;
    const int coordinates[]{key.x, key.y, key.z};

    for (auto coordinate : coordinates)
        valid = valid && std::int64_t(coordinate) * 8 >= INT_MIN && std::int64_t(coordinate) * 8 + 7 <= INT_MAX;

    if (i) {
        const auto old = keys[i - 1];
        valid = valid && (old.z < key.z || (old.z == key.z && (old.y < key.y || (old.y == key.y && old.x < key.x))));
    }
    if (!valid) atomicOr(status, 1U);
}
__global__ void nearest_kernel(tsdf_view field, device_index_view index, sensor_grid::field_view source,
    const block_key* keys, unsigned count, detail::cuda_projective_pose transform, float size, float cap, float band, unsigned* status)
{
    const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= std::size_t(count) * 512 || *status) return;
    const auto key = keys[i / 512];
    const auto local = unsigned(i % 512);
    const auto slot = find_block(index, key).slot;
    const float3 point{(key.x * 8 + int(local % 8) + .5f) * size,
        (key.y * 8 + int(local / 8 % 8) + .5f) * size, (key.z * 8 + int(local / 64) + .5f) * size};
    const auto sensor = detail::projective_reference_point(transform, point);
    const auto measured = sensor_grid::sample(source, sensor.x, sensor.y, sensor.z, sensor_grid::interpolation::nearest);
    if (!measured.valid) return;
    auto value = field[slot].voxels[local];
    const float old = value.flags & 1U ? value.weight : 0.f;
    const double total = double(old) + measured.weight;
    value.distance = float((double(old) * (old > 0 ? value.distance : 0.f) + double(measured.weight) * measured.distance) / total);
    value.weight = float(fmin(double(cap), total));
    value.flags = 1U | (fabsf(value.distance) <= band ? 2U : 0U);
    if (value.sample_count != UINT32_MAX) ++value.sample_count;
}
bool valid_mapping(storage::device_descriptor field, device_index_view index, unsigned* status)
{
    if (field.record_schema != storage::schema::tsdf || field.size_tiles > field.capacity_tiles || !field.generation ||
        !status || !index.entries || !index.capacity || (index.capacity & (index.capacity - 1)) ||
        !index.allocation_generation || index.field != 0U || field.device != index.device) return false;
    int current{-1};
    if (cudaGetDevice(&current) != cudaSuccess || current != field.device || !field.tile_stride ||
        field.tile_stride % 4 || field.voxel_stride % 4 || field.voxel_stride < 4 ||
        unsigned(field.storage_layout) > unsigned(storage::layout::pool_soa)) return false;

    for (unsigned f = 0; f < 4; ++f)
        if (!field.fields[f] || reinterpret_cast<std::uintptr_t>(field.fields[f]) % 4) return false;
    return true;
}
int integrate_ordered(storage::device_descriptor payload, device_index_view index,
    const detail::cuda_ray_contribution* samples, std::size_t sample_count,
    const detail::cuda_voxel_segment* segments, std::uint32_t count,
    detail::cuda_fusion_parameters config, unsigned* status, void* stream)
{
    if (!valid_mapping(payload, index, status) || !samples || !segments || !count || count > 16777216 ||
        !isfinite(config.truncation_distance) || config.truncation_distance <= 0 ||
        !isfinite(config.maximum_weight) || config.maximum_weight <= 0 ||
        !isfinite(config.extraction_distance) || config.extraction_distance < 0) return int(cudaErrorInvalidValue);
    const auto native = static_cast<cudaStream_t>(stream);
    auto error = cudaMemsetAsync(status, 0, sizeof(unsigned), native);
    if (error) return int(error);
    const tsdf_view field{storage::device_accessor<storage::schema::tsdf>{payload}};
    ordered_kernel<true><<<(count + 127) / 128, 128, 0, native>>>(field, index, samples, sample_count, segments, count, config, status);
    error = cudaGetLastError();
    if (error) return int(error);
    ordered_kernel<false><<<(count + 127) / 128, 128, 0, native>>>(field, index, samples, sample_count, segments, count, config, status);
    return int(cudaGetLastError());
}
int integrate_nearest(storage::device_descriptor payload, device_index_view index,
    sensor_grid::field_view source, const block_key* keys, std::uint32_t count,
    detail::cuda_projective_pose transform, float size, float weight, float band, unsigned* status, void* stream)
{
    if (!valid_mapping(payload, index, status) || !sensor_grid::sampling_detail::valid_view(source) || !keys || !count ||
        count > 1048576 || !isfinite(size) || size <= 0 || !isfinite(weight) || weight <= 0 || !isfinite(band) || band < 0)
        return int(cudaErrorInvalidValue);
    double norm{};

    for (auto component : transform.rotation_xyzw) {
        if (!isfinite(component)) return int(cudaErrorInvalidValue);
        norm += double(component) * component;
    }
    for (auto component : transform.translation)
        if (!isfinite(component)) return int(cudaErrorInvalidValue);
    if (fabs(norm - 1.0) > 1.0e-5) return int(cudaErrorInvalidValue);
    const auto native = static_cast<cudaStream_t>(stream);
    auto error = cudaMemsetAsync(status, 0, sizeof(unsigned), native);
    if (error) return int(error);
    validate_keys<<<(count + 127) / 128, 128, 0, native>>>(index, keys, count, payload.size_tiles, status);
    error = cudaGetLastError();
    if (error) return int(error);
    nearest_kernel<<<count * 4, 128, 0, native>>>(tsdf_view{storage::device_accessor<storage::schema::tsdf>{payload}},
        index, source, keys, count, transform, size, weight, band, status);
    return int(cudaGetLastError());
}
}
