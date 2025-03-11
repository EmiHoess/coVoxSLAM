#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace covoxslam::storage
{
inline constexpr std::size_t voxels_per_tile = 512;
enum class schema { tsdf, esdf, blocks_esdf, projective_esdf };
enum class layout { pooled_aos, tile_aosoa, pool_soa };
struct tsdf_value
{
    float distance{}, weight{};
    std::uint32_t sample_count{}, flags{};
};
struct esdf_value {
    float signed_distance_m{};
    std::int32_t parent_dx{}, parent_dy{}, parent_dz{};
    std::uint32_t flags{};
};
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
template<schema s> inline constexpr std::size_t field_count = schema_traits<s>::fields;
template<schema s, std::size_t field>
using field_type = std::conditional_t<s == schema::tsdf,
    std::conditional_t<(field < 2), float, std::uint32_t>,
    std::conditional_t<field == 0, float, std::conditional_t<field == 4, std::uint32_t, std::int32_t>>>;
template<schema s> struct aos_tile { std::array<value<s>, voxels_per_tile> voxels{}; };
}
