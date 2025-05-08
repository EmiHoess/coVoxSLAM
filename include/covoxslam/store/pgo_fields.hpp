#pragma once
#include <covoxslam/store/geometry.hpp>
#include <covoxslam/storage/transfer_plan.hpp>
#include <covoxslam/pgo.hpp>
#if defined(covoxslam_has_pgo)
#include <ceronoceres/field_views.hpp>

namespace covoxslam::store
{
enum class pgo_field_mode { copied, borrowed_cpu, resident_cuda };
struct pgo_field_binding
{
    std::uint64_t submap{};
    std::size_t volume{};
     epochs expected;
};
template<storage::schema s> ceronoceres::field_tile_view pgo_tile(
    storage::transfer_endpoint source, const slot_record& slot, std::size_t index,
    std::size_t volume, std::uint64_t content, std::uint64_t generation, geometry_parameters parameters)
{
    ceronoceres::field_tile_view tile;
    tile.volume = volume;
    tile.x = slot.key.x; tile.y = slot.key.y; tile.z = slot.key.z;
    tile.kind = s == storage::schema::tsdf ? ceronoceres::FieldKind::Tsdf : ceronoceres::FieldKind::Esdf;
    tile.distance = source.fields[0] + index * source.tile_stride;
    tile.flags = source.fields[storage::field_count<s> - 1] + index * source.tile_stride;
    if constexpr (s == storage::schema::tsdf) tile.weight = source.fields[1] + index * source.tile_stride;
    if constexpr (s == storage::schema::projective_esdf) tile.encoding = ceronoceres::field_encoding::squared_voxels;
    tile.voxel_stride = source.voxel_stride;
    tile.voxel_size = parameters.voxel_size;
    tile.maximum_distance = parameters.distance_limit;
    tile.content_epoch = content; tile.allocation_generation = generation;
    tile.slot_generation = slot.generation;
    return tile;
}
}
#endif
