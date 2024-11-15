#include "cuda_resident_internal.hpp"
#include <bit>

namespace covoxslam
{
cuda_resident_context::cuda_resident_context() = default;
cuda_resident_context::~cuda_resident_context() = default;
std::size_t cuda_resident_context::device_bytes() const noexcept { return state->bytes; }

operation_result<std::unique_ptr<cuda_resident_context>> cuda_resident_context::create(
    mapping_config config, cuda_resident_options options)
{

    if (!config.valid() || !options.tile_capacity || options.tile_capacity > UINT32_MAX / 512 ||
        options.tile_capacity > config.maximum_tiles || !options.ray_capacity || !options.contribution_capacity)
        return std::unexpected(operation_error{"invalid resident mapping configuration or capacities"});
    const auto voxels = static_cast<std::size_t>(options.tile_capacity) * 512;
    const auto buckets = std::bit_ceil(options.tile_capacity * 2U);
    const auto sums = (voxels + 255) / 256;
    const auto bytes = sizeof(detail::cuda_resident_state) + std::size_t(buckets) * sizeof(detail::cuda_resident_hash_entry) +
        options.tile_capacity * sizeof(detail::cuda_tile_coordinate) + voxels * (2 * sizeof(detail::cuda_tsdf_voxel) + 16) +
        sums * sizeof(std::uint64_t) + options.contribution_capacity * sizeof(detail::cuda_resident_contribution) +
        options.ray_capacity * sizeof(detail::cuda_device_ray);

    if (options.maximum_device_bytes && bytes > options.maximum_device_bytes)
        return std::unexpected(operation_error{"resident storage exceeds the device byte budget"});
    const auto device = query_cuda_device_checked(options.device);
    if (!device) return std::unexpected(device.error());

    if (bytes > device->total_memory_bytes)
        return std::unexpected(operation_error{"resident storage exceeds total device memory"});
    auto owner = std::unique_ptr<cuda_resident_context>(new cuda_resident_context);
    owner->state = std::make_unique<implementation>();
    auto& data = *owner->state;
    data.config = config;
     data.options = options;
    data.bytes = bytes;
    auto status = data.check(cudaSetDevice(options.device));

    if (!status) return std::unexpected(status.error());
    status = data.check(cudaStreamCreateWithFlags(&data.stream, cudaStreamNonBlocking));
    if (!status) return std::unexpected(status.error());
    auto& view = data.view;
    view.tile_capacity = options.tile_capacity;
    view.hash_capacity = buckets;
    view.contribution_capacity = options.contribution_capacity;
    status = data.allocate(view.state, 1);

    if (status) status = data.allocate(view.hash, buckets);
    if (status) status = data.allocate(view.keys, options.tile_capacity);
    if (status) status = data.allocate(view.tiles, options.tile_capacity);

    if (status) status = data.allocate(view.pending, voxels);
    if (status) status = data.allocate(view.counts, voxels);

    if (status) status = data.allocate(view.cursors, voxels);
    if (status) status = data.allocate(view.offsets, voxels);

    if (status) status = data.allocate(view.block_sums, sums);
    if (status) status = data.allocate(view.contributions, options.contribution_capacity);

    if (status) status = data.allocate(data.rays, options.ray_capacity);
    if (status) status = data.check(detail::cuda_initialize_resident_map(view, data.stream));

    if (status) status = data.check(cudaStreamSynchronize(data.stream));
    if (!status) return std::unexpected(status.error());
    return owner;
}
}
