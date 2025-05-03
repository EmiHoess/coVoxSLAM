#pragma once
#include <covoxslam/storage/schema.hpp>
#include <covoxslam/result.hpp>
#include <cstdint>

namespace covoxslam::store
{
enum class distance_units { meters, squared_voxels };
enum class parent_coordinates { relative, absolute };
struct esdf_contract
{
    storage::schema field;
    distance_units distance;
    parent_coordinates parent;
    bool signed_seed;
};
inline operation_result<esdf_contract> describe_esdf(storage::schema field)
{
    switch (field) {
    case storage::schema::esdf: return esdf_contract{field, distance_units::meters, parent_coordinates::relative, false};
    case storage::schema::blocks_esdf: return esdf_contract{field, distance_units::meters, parent_coordinates::relative, true};
    case storage::schema::projective_esdf: return esdf_contract{field, distance_units::squared_voxels, parent_coordinates::absolute, false};
    default: return std::unexpected(operation_error{"field is not an ESDF schema"});
    }
}
struct esdf_dependency
{
    std::uint64_t submap{}, tsdf_epoch{}, esdf_epoch{}, source_epoch{};
    bool current() const noexcept { return tsdf_epoch && esdf_epoch && source_epoch == tsdf_epoch; }
};
}
