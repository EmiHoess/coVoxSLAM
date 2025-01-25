#include "sensor_grid_map_internal.hpp"

namespace covoxslam::sensor_grid
{
gpu_map::gpu_map() = default;
gpu_map::~gpu_map() = default;
operation_result<std::unique_ptr<gpu_map>> gpu_map::create(const lookup& table,
    std::uint32_t capacity, int device, std::size_t maximum_bytes, transfer_options options)
{
    auto status = validate_lookup(table);

    if (!status) return std::unexpected(status.error());
    const std::size_t side = (table.grid.side + 7) / 8, count = side * side * side;
    compact_mask_layout compact;
    if (options.source_mask == source_mask_mode::compact) {
        auto built = build_compact_mask(table);
        if (!built) return std::unexpected(built.error());
        compact = std::move(*built);
    }
    const bool masked = options.candidates != candidate_mode::whole_tiles;
    const bool listed = options.candidates == candidate_mode::compact_fp32;
    const std::size_t list_bytes = listed ? std::size_t(options.candidate_capacity) * 4 + sizeof(candidate_list_counters) : 0;
    const std::size_t candidate_bytes = list_bytes + (masked ? std::size_t(capacity) * 64 + table.active_voxels.size() * 4 + 8 : 0);
    const std::size_t bytes = candidate_bytes + std::size_t(capacity) * (sizeof(detail::cuda_tsdf_tile) + 16) + count + 8 +
        (compact.offsets.size() + compact.cells.size()) * 4;
    if (!capacity || capacity > 1048576 || device < 0 || bytes > maximum_bytes ||
        unsigned(options.source_mask) > 1 || unsigned(options.arithmetic) > 1 || unsigned(options.candidates) > 3 ||
        (listed && (!options.candidate_capacity || options.candidate_capacity > 536870912)) ||
        (masked && (options.specialized || options.arithmetic != transfer_arithmetic::fp64)))
        return std::unexpected(operation_error{"sensor TSDF exceeds storage budget"});
    auto owner = std::unique_ptr<gpu_map>(new gpu_map);
    owner->state = std::make_unique<implementation>();
    auto& data = *owner->state;
    data.settings = table.settings;
    data.options = options;
    data.active_count = static_cast<std::uint32_t>(table.active_voxels.size());
     data.grid = table.grid;
    data.capacity = capacity;
    data.device = device;
    data.flags.resize(count);
    status = data.check(cudaSetDevice(device));
    if (!status) return std::unexpected(status.error());
    data.selected = true;
    status = data.check(cudaStreamCreateWithFlags(&data.stream, cudaStreamNonBlocking));
    if (status) status = data.allocate(data.tiles, capacity);
    if (status) status = data.allocate(data.keys, capacity);
    if (status) status = data.allocate(data.slots, capacity);
    if (status) status = data.allocate(data.mask, count);
    if (status) status = data.allocate(data.updates, 1);
    if (status && masked) status = data.allocate(data.candidate_masks, std::size_t(capacity) * 16);
    if (status && masked) status = data.allocate(data.active_cells, table.active_voxels.size());
    if (status && masked) status = data.allocate(data.candidate_count, 1);
    if (status && listed) status = data.allocate(data.candidate_list.ids, options.candidate_capacity);
    if (status && listed) status = data.allocate(data.candidate_list.counters, 1);
    data.candidate_list.masks = data.candidate_masks;
    data.candidate_list.capacity = options.candidate_capacity;
    if (status && masked) status = data.check(cudaMemcpyAsync(data.active_cells, table.active_voxels.data(),
        table.active_voxels.size() * 4, cudaMemcpyHostToDevice, data.stream));
    if (status && !compact.offsets.empty()) status = data.allocate(data.mask_offsets, compact.offsets.size());
    if (status && !compact.cells.empty()) status = data.allocate(data.mask_cells, compact.cells.size());
    if (status && data.mask_offsets) status = data.check(cudaMemcpyAsync(data.mask_offsets, compact.offsets.data(),
        compact.offsets.size() * 4, cudaMemcpyHostToDevice, data.stream));
    if (status && data.mask_cells) status = data.check(cudaMemcpyAsync(data.mask_cells, compact.cells.data(),
        compact.cells.size() * 4, cudaMemcpyHostToDevice, data.stream));

    for (auto& event : data.events)
        if (status) status = data.check(cudaEventCreate(&event));
    if (status) status = data.check(cudaMemsetAsync(data.tiles, 0, std::size_t(capacity) * sizeof(*data.tiles), data.stream));
    status = data.wait(status);
    if (!status) return std::unexpected(status.error());
    return owner;
}
operation_status gpu_map::reset()
{
    auto& data = *state;

    if (data.failed) return std::unexpected(operation_error{"sensor TSDF is in failed state"});
    auto status = data.check(cudaSetDevice(data.device));
    if (status) status = data.check(cudaMemsetAsync(data.tiles, 0, std::size_t(data.capacity) * sizeof(*data.tiles), data.stream));
    status = data.wait(status);
    if (!status) return status;
    data.index.clear();
     data.indices.clear();
    return {};
}
operation_result<tsdf_tiles> gpu_map::download()
{
    auto& data = *state;
    if (data.failed) return std::unexpected(operation_error{"cannot publish failed sensor TSDF"});
    std::vector<detail::cuda_tsdf_tile> staging(data.indices.size());
    auto status = data.check(cudaSetDevice(data.device));
    if (status && !staging.empty()) status = data.check(cudaMemcpyAsync(staging.data(), data.tiles,
        staging.size() * sizeof(*data.tiles), cudaMemcpyDeviceToHost, data.stream));
    status = data.wait(status);
    if (!status) return std::unexpected(status.error());
    tsdf_tiles output;
    output.reserve(staging.size());

    for (std::size_t i = 0; i < staging.size(); ++i)
    {
        auto& tile = output[data.indices[i]];

        for (std::size_t j = 0; j < tile_voxels; ++j) {
            const auto& value = staging[i].voxels[j];
             tile.voxels[j] = {value.distance, value.weight, value.sample_count, value.flags};
        }
    }
    return output;
}
}
