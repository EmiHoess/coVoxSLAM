#pragma once
#include "discovery.h"
#include "../ray_resident/types.h"

namespace covoxslam::sensor_grid
{
struct resident_discovery_view
{
    discovery_view scan{};
    detail::cuda_resident_hash_entry* hash{};
    detail::cuda_tile_coordinate* keys{};
    std::uint64_t* scratch_keys{}, *offsets{}, *sums{};
    std::uint32_t* missing{}, *slots{};
};
int cuda_sensor_allocate_tiles(resident_discovery_view view, void* stream);
int cuda_sensor_reset_tiles(resident_discovery_view view, detail::cuda_tsdf_tile* tiles, void* stream);
int cuda_sensor_restore_tiles(resident_discovery_view view, std::uint32_t count, void* stream);
int cuda_sensor_resident_transfer(field_view source, interpolation mode, resident_discovery_view view,
    detail::cuda_tsdf_tile* tiles, detail::cuda_projective_pose transform, float size,
    float maximum_weight, float band, unsigned long long* updates, void* stream);
inline bool valid_resident_discovery(resident_discovery_view view) noexcept
{
    return view.scan.hash && view.scan.keys && view.scan.state && view.hash && view.keys &&
        view.scratch_keys && view.offsets && view.sums && view.missing && view.slots &&
        view.scan.capacity && view.scan.capacity <= 1048576 &&
        view.scan.hash_capacity == discovery_hash_size(view.scan.capacity);
}
}
