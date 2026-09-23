#include <covoxslam/mapping/esdf/band_esdf.hpp>
#include "band_initialize.cuh"
#include "band_propagate.cuh"
#include "band_publish.cuh"
#include "detail/timing.cuh"

namespace covoxslam::store
{
namespace
{
inline constexpr unsigned tile_kernel_threads = 128;
inline constexpr unsigned tile_kernel_blocks = unsigned(tile_voxels) / tile_kernel_threads;
inline constexpr unsigned sweep_tiles_per_block = tile_kernel_threads / unsigned(tile_face_voxels);
static_assert(tile_voxels % tile_kernel_threads == 0);
static_assert(tile_kernel_threads % tile_face_voxels == 0);
}
int run_band_esdf(storage::device_descriptor source, device_index_view source_index,
    storage::device_descriptor target, device_index_view target_index,
    storage::device_descriptor work, const block_key* keys, unsigned count,
    detail::cuda_tile_coordinate* coordinates, band_tile_slot* slots, band_status* status,
    band_esdf_options options, band_esdf_metrics* metrics, void* selected)
{
    if (!options.valid() || !count || !keys || !coordinates || !slots || !status || !metrics ||
        !band_descriptor_valid(source, storage::schema::tsdf) ||
        !band_descriptor_valid(target, storage::schema::esdf) ||
        !band_descriptor_valid(work, storage::schema::projective_esdf) || work.size_tiles != count ||
        !band_index_valid(source_index, source, 0) || !band_index_valid(target_index, target, 1) ||
        source_index.submap != target_index.submap || work.device != source.device ||
        !band_disjoint(source, target) || !band_disjoint(source, work) || !band_disjoint(target, work))
        return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(selected);
    const band_tsdf_view input{storage::device_accessor<storage::schema::tsdf, true>{source}};
    const band_work_view output{storage::device_accessor<storage::schema::projective_esdf>{work}};
    const float extent = options.field.maximum_distance_m / options.field.voxel_size;
    const float squared = extent * extent;
    band_event_timer timer;
    auto timed = timer.start(stream);
    if (timed) return int(timed);
    auto error = cudaMemsetAsync(status, 0, sizeof(*status), stream);
    if (error) return int(error);
    index_band_tiles<<<(count + 127) / 128, 128, 0, stream>>>(keys, count, source_index,
        target_index, source.size_tiles, target.size_tiles, coordinates, slots, status);
    error = cudaGetLastError();

    if (error) return int(error);
    unsigned invalid{};
    error = timer.read(&status->invalid, invalid, stream, *metrics);
    if (error || invalid) return int(error ? error : cudaErrorInvalidValue);
    error = timer.start(stream);
    if (error) return int(error);
    initialize_band_fields<<<count * tile_kernel_blocks, tile_kernel_threads, 0, stream>>>(
        input, output, slots, coordinates,
        count, options.field, squared, status);
    error = cudaGetLastError();

    if (error) return int(error);
    error = timer.read(&status->invalid, invalid, stream, *metrics);
     if (error || invalid) return int(error ? error : cudaErrorInvalidValue);
    metrics->rounds = 0;
    unsigned changed{band_changed_mask};
    const auto propagation_started = std::chrono::steady_clock::now();

    while (changed && metrics->rounds < options.maximum_rounds)
    {
        error = timer.start(stream);
        if (error) return int(error);
        error = cudaMemsetAsync(&status->changed, 0, sizeof(status->changed), stream);
        if (error) return int(error);

        for (int axis = 0; axis < 3; ++axis) {
            sweep_band_axis<<<(count + sweep_tiles_per_block - 1) / sweep_tiles_per_block,
                tile_kernel_threads, 0, stream>>>(coordinates, count,
                output, axis, squared, status);
            error = cudaGetLastError();
            if (error) return int(error);
        }

        for (int parity = 0; parity < 2; ++parity)
        {
            relax_band_neighbors<<<count * tile_kernel_blocks, tile_kernel_threads, 0, stream>>>(
                coordinates, count,
                output, squared, parity, status);
             error = cudaGetLastError();
            if (error) return int(error);
        }
        error = timer.read(&status->changed, changed, stream, *metrics);
        if (error) return int(error);
        ++metrics->rounds;
    }
    if (changed) return int(cudaErrorNotReady);
    const auto publication_started = std::chrono::steady_clock::now();
    metrics->propagation_ms = std::chrono::duration<double, std::milli>(publication_started - propagation_started).count();
    error = timer.start(stream);
    if (error) return int(error);
    const band_output_view destination{storage::device_accessor<storage::schema::esdf>{target}};
    if (target.size_tiles) {
        clear_band_output<<<unsigned(target.size_tiles * tile_kernel_blocks),
            tile_kernel_threads, 0, stream>>>(destination);
        error = cudaGetLastError();
        if (error) return int(error);
    }
    publish_band_fields<<<count * tile_kernel_blocks, tile_kernel_threads, 0, stream>>>(
        output, destination, slots, coordinates,
        count, options.field.voxel_size, options.field.maximum_distance_m, status);
    error = cudaGetLastError();
    if (error) return int(error);
    error = timer.read(&status->invalid, invalid, stream, *metrics);
    metrics->publication_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - publication_started).count();
    return int(error ? error : invalid ? cudaErrorInvalidValue : cudaSuccess);
}
}
