#pragma once
#include <covoxslam/voxel/records.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace covoxslam::storage
{
inline constexpr std::size_t voxels_per_tile = 512;
enum class schema { tsdf = 0, esdf = 1, blocks_esdf = 2, projective_esdf = 3 };
enum class layout { pooled_aos = 0, tile_aosoa = 1, pool_soa = 2 };
using tsdf_value = tsdf_record;
using esdf_value = esdf_record;
struct blocks_esdf_value
{
    float distance_m{};
    std::int32_t parent_dx{}, parent_dy{}, parent_dz{};
    std::uint32_t flags{};
};
struct projective_esdf_value
{
    float squared_distance_voxels{};
    std::int32_t parent_x{}, parent_y{}, parent_z{};
     std::uint32_t flags{};
};
template<schema s> struct schema_traits;
template<> struct schema_traits<schema::tsdf> { using value_type = tsdf_value; static constexpr std::size_t fields = 4; };
template<> struct schema_traits<schema::esdf> { using value_type = esdf_value; static constexpr std::size_t fields = 5; };
template<> struct schema_traits<schema::blocks_esdf> { using value_type = blocks_esdf_value; static constexpr std::size_t fields = 5; };
template<> struct schema_traits<schema::projective_esdf> { using value_type = projective_esdf_value; static constexpr std::size_t fields = 5; };
template<schema s> using value = typename schema_traits<s>::value_type;
template<schema s> inline constexpr std::size_t record_bytes = sizeof(value<s>);
template<schema s> inline constexpr std::size_t canonical_tile_bytes = voxels_per_tile * record_bytes<s>;
template<schema s> inline constexpr std::size_t field_count = schema_traits<s>::fields;
template<schema s, std::size_t field>
using field_type = std::conditional_t<s == schema::tsdf,
    std::conditional_t<(field < 2), float, std::uint32_t>,
    std::conditional_t<field == 0, float, std::conditional_t<field == 4, std::uint32_t, std::int32_t>>>;
template<schema s> struct aos_tile { std::array<value<s>, voxels_per_tile> voxels{}; };
}
