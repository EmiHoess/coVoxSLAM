#include "active_submap_internal.hpp"

namespace covoxslam::sensor_grid
{
gpu_map::gpu_map() = default;
gpu_map::~gpu_map() = default;
operation_result<std::unique_ptr<gpu_map>> gpu_map::create(const lookup& table,
    std::uint32_t capacity, int device, std::size_t maximum_bytes, transfer_options options, bool resident_soa)
{
    auto status = validate_lookup(table);

    if (!status) return std::unexpected(status.error());
    if (!resident_soa || options.discovery != discovery_mode::gpu_resident ||
        options.source_mask != source_mask_mode::dense || options.candidates != candidate_mode::whole_tiles ||
        options.specialized || options.arithmetic != transfer_arithmetic::fp64)
        return std::unexpected(operation_error{"coVoxSLAM requires resident SoA reference transfer"});
    const std::size_t side = (table.grid.side + 7) / 8, count = side * side * side;
    const auto hash_capacity = discovery_hash_size(capacity);
    const std::size_t discovery_bytes = (std::size_t(hash_capacity) + capacity) * 8 + sizeof(discovery_state);
    const std::size_t resident_bytes = std::size_t(hash_capacity) * 16 +
        std::size_t(capacity) * 20 + ((std::size_t(capacity) + 255) / 256) * 8;
    const std::size_t bytes = resident_bytes + discovery_bytes +
        std::size_t(capacity) * (512 * sizeof(detail::cuda_tsdf_voxel) + 16) + count + 8;

    if (!capacity || capacity > 1048576 || device < 0 || bytes > maximum_bytes)
        return std::unexpected(operation_error{"sensor TSDF exceeds storage budget"});
    auto owner = std::unique_ptr<gpu_map>(new gpu_map);
    owner->state = std::make_unique<implementation>();
    auto& data = *owner->state;
    data.settings = table.settings;
    data.options = options;
    data.soa_selected = resident_soa;
     data.grid = table.grid;
    data.capacity = capacity;
    data.device = device;
    status = data.check(cudaSetDevice(device));
    if (!status) return std::unexpected(status.error());
    data.selected = true;
    status = data.check(cudaStreamCreateWithFlags(&data.stream, cudaStreamNonBlocking));
    if (status) {
        data.soa = std::make_unique<storage::device_buffer<storage::schema::tsdf, storage::layout::pool_soa>>();
        status = data.soa->initialize(capacity);
    }
    if (status) status = data.allocate(data.keys, capacity);
    if (status) status = data.allocate(data.slots, capacity);
    if (status) status = data.allocate(data.mask, count);
    if (status) status = data.allocate(data.updates, 1);
    if (status) status = data.allocate(data.discovery.hash, hash_capacity);
    if (status) status = data.allocate(data.discovery.keys, capacity);
    if (status) status = data.allocate(data.discovery.state, 1);
    data.discovery.capacity = capacity;
    data.discovery.hash_capacity = hash_capacity;
    if (status) status = data.check(cudaMemsetAsync(data.discovery.state, 0, sizeof(discovery_state), data.stream));
    data.resident.scan = data.discovery;
    data.resident.keys = data.keys; data.resident.slots = data.slots;
    if (status) status = data.allocate(data.resident.hash, hash_capacity);
    if (status) status = data.allocate(data.resident.scratch_keys, capacity);
    if (status) status = data.allocate(data.resident.offsets, capacity);
    if (status) status = data.allocate(data.resident.sums, (capacity + 255) / 256);
    if (status) status = data.allocate(data.resident.missing, capacity);

    for (auto& event : data.events)
        if (status) status = data.check(cudaEventCreate(&event));
    if (status) status = data.reset_payload();
    status = data.wait(status);
    if (!status) return std::unexpected(status.error());
    return owner;
}
operation_status gpu_map::reset()
{
    auto& data = *state;

    if (data.failed) return std::unexpected(operation_error{"sensor TSDF is in failed state"});
    auto status = data.check(cudaSetDevice(data.device));
    if (status) status = data.reset_payload();
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
    return data.download_soa();
}
}
