#include <covoxslam/store/band_esdf.hpp>
#include "band_esdf_initialize.cuh"
#include "band_esdf_sweep.cuh"
#include "band_esdf_neighbors.cuh"
#include "band_esdf_publish.cuh"

namespace covoxslam::store
{
cudaError_t read_band_status(unsigned* pointer, unsigned& value, cudaStream_t stream)
{
    const auto copied = cudaMemcpyAsync(&value, pointer, sizeof(value), cudaMemcpyDeviceToHost, stream);
    const auto waited = cudaStreamSynchronize(stream);
    return copied == cudaSuccess ? waited : copied;
}
int run_band_esdf(storage::device_descriptor source, device_index_view source_index,
    storage::device_descriptor target, device_index_view target_index,
    storage::device_descriptor work, const block_key* keys, unsigned count,
    detail::cuda_tile_coordinate* coordinates, void* slot_buffer, unsigned* status,
    band_esdf_options options, band_esdf_metrics* metrics, void* selected)
{
    if (!options.valid() || !count || !keys || !coordinates || !slot_buffer || !status || !metrics ||
        !band_descriptor_valid(source, storage::schema::tsdf) ||
        !band_descriptor_valid(target, storage::schema::esdf) ||
        !band_descriptor_valid(work, storage::schema::projective_esdf) || work.size_tiles != count ||
        !band_index_valid(source_index, source, 0) || !band_index_valid(target_index, target, 1) ||
        source_index.submap != target_index.submap || work.device != source.device ||
        !band_disjoint(source, target) || !band_disjoint(source, work) || !band_disjoint(target, work))
        return int(cudaErrorInvalidValue);
    const auto stream = static_cast<cudaStream_t>(selected);
    auto* slots = static_cast<band_tile_slot*>(slot_buffer);
    const band_tsdf_view input{storage::device_accessor<storage::schema::tsdf, true>{source}};
    const band_work_view output{storage::device_accessor<storage::schema::projective_esdf>{work}};
    const float extent = options.field.maximum_distance_m / options.field.voxel_size;
    const float squared = extent * extent;
    auto error = cudaMemsetAsync(status, 0, 8, stream);
    if (error) return int(error);
    index_band_tiles<<<(count + 127) / 128, 128, 0, stream>>>(keys, count, source_index,
        target_index, source.size_tiles, target.size_tiles, coordinates, slots, status);
    error = cudaGetLastError();

    if (error) return int(error);
    unsigned invalid{};
    error = read_band_status(status, invalid, stream);
    if (error || invalid) return int(error ? error : cudaErrorInvalidValue);
    initialize_band_fields<<<count * 4, 128, 0, stream>>>(input, output, slots, coordinates,
        count, options.field, squared, status);
    error = cudaGetLastError();

    if (error) return int(error);
    error = read_band_status(status, invalid, stream);
     if (error || invalid) return int(error ? error : cudaErrorInvalidValue);
    metrics->rounds = 0;
    unsigned changed{1};

    while (changed && metrics->rounds < options.maximum_rounds)
    {
        error = cudaMemsetAsync(status + 1, 0, 4, stream);
        if (error) return int(error);

        for (int axis = 0; axis < 3; ++axis) {
            sweep_band_axis<<<(count + 1) / 2, 128, 0, stream>>>(coordinates, count,
                output, axis, squared, status + 1);
            error = cudaGetLastError();
            if (error) return int(error);
        }

        for (int parity = 0; parity < 2; ++parity)
        {
            relax_band_neighbors<<<count * 4, 128, 0, stream>>>(coordinates, count,
                output, squared, parity, status + 1);
             error = cudaGetLastError();
            if (error) return int(error);
        }
        error = read_band_status(status + 1, changed, stream);
        if (error) return int(error);
        ++metrics->rounds;
    }
    if (changed) return int(cudaErrorNotReady);
    const band_output_view destination{storage::device_accessor<storage::schema::esdf>{target}};
    if (target.size_tiles) {
        clear_band_output<<<unsigned(target.size_tiles * 4), 128, 0, stream>>>(destination);
        error = cudaGetLastError();
        if (error) return int(error);
    }
    publish_band_fields<<<count * 4, 128, 0, stream>>>(output, destination, slots, coordinates,
        count, options.field.voxel_size, options.field.maximum_distance_m, status);
    error = cudaGetLastError();
    if (error) return int(error);
    error = read_band_status(status, invalid, stream);
    return int(error ? error : invalid ? cudaErrorInvalidValue : cudaSuccess);
}
}
