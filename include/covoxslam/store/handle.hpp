#pragma once
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/schema.hpp>
#include <covoxslam/map_store/key.hpp>
#include <covoxslam/core/result.hpp>
#include <limits>

namespace covoxslam::store
{
enum class field_kind : std::uint32_t { tsdf, esdf };
enum class lifecycle : std::uint32_t { active, finalized, retired };
enum class residency : std::uint32_t { absent, resident, evicting, evicted, restoring };
enum class writer_kind : std::uint32_t { none, cpu, gpu };
struct tile_handle
{
    std::uint64_t submap{};
    field_kind field{};
    std::uint32_t slot{std::numeric_limits<std::uint32_t>::max()};
     std::uint64_t reuse_generation{};
    bool operator==(const tile_handle&) const = default;
};
struct epochs {
    std::uint64_t descriptor{1}, allocation{1}, tsdf{}, esdf{}, esdf_source{}, pose{1};
};
struct layout_version
{
    storage::schema schema{};
    storage::layout layout{};
    std::uint32_t format{1}, voxels_per_tile{512};
    bool operator==(const layout_version&) const = default;
};
struct pose_record
{
    double x{}, y{}, z{}, yaw{};
    bool operator==(const pose_record&) const = default;
};
struct submap_record {
    std::uint64_t id{};
    lifecycle state{lifecycle::active};
    residency cpu{residency::absent}, gpu{residency::absent};
    writer_kind writer{writer_kind::none};
    epochs versions;
    pose_record initial_pose, optimized_pose;
};
inline operation_result<std::uint64_t> next_epoch(std::uint64_t current)
{
    if (current == std::numeric_limits<std::uint64_t>::max())
        return std::unexpected(operation_error{"store epoch is exhausted"});
    return current + 1;
}
}
