#pragma once
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/voxel/esdf_encodings.hpp>
#include <covoxslam/voxel/records.hpp>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace covoxslam::storage
{
enum class schema { tsdf = 0, esdf = 1, blocks_esdf = 2, projective_esdf = 3 };
static_assert(static_cast<int>(schema::tsdf) == 0 && static_cast<int>(schema::esdf) == 1 &&
    static_cast<int>(schema::blocks_esdf) == 2 && static_cast<int>(schema::projective_esdf) == 3);
using tsdf_value = tsdf_record;
using esdf_value = esdf_record;
template<schema s> struct schema_traits;
template<> struct schema_traits<schema::tsdf> { using value_type = tsdf_value; static constexpr std::size_t fields = 4; };
template<> struct schema_traits<schema::esdf> { using value_type = esdf_value; static constexpr std::size_t fields = 5; };
template<> struct schema_traits<schema::blocks_esdf> { using value_type = blocks_esdf_value; static constexpr std::size_t fields = 5; };
template<> struct schema_traits<schema::projective_esdf> { using value_type = projective_esdf_value; static constexpr std::size_t fields = 5; };
template<schema s> using value = typename schema_traits<s>::value_type;
template<schema s> inline constexpr std::size_t record_bytes = sizeof(value<s>);
template<schema s> inline constexpr std::size_t canonical_tile_bytes = voxels_per_tile * record_bytes<s>;
template<schema s> inline constexpr std::size_t field_count = schema_traits<s>::fields;
static_assert(field_count<schema::tsdf> == 4 && field_count<schema::esdf> == 5 &&
    field_count<schema::blocks_esdf> == 5 && field_count<schema::projective_esdf> == 5);
template<schema s, std::size_t field>
using field_type = std::conditional_t<s == schema::tsdf,
    std::conditional_t<(field < 2), float, std::uint32_t>,
    std::conditional_t<field == 0, float, std::conditional_t<field == 4, std::uint32_t, std::int32_t>>>;
static_assert(std::is_same_v<field_type<schema::tsdf, 0>, float> &&
    std::is_same_v<field_type<schema::tsdf, 1>, float> &&
    std::is_same_v<field_type<schema::tsdf, 2>, std::uint32_t> &&
    std::is_same_v<field_type<schema::tsdf, 3>, std::uint32_t>);
static_assert(std::is_same_v<field_type<schema::esdf, 0>, float> &&
    std::is_same_v<field_type<schema::esdf, 1>, std::int32_t> &&
    std::is_same_v<field_type<schema::esdf, 4>, std::uint32_t>);
static_assert(std::is_same_v<field_type<schema::blocks_esdf, 0>, float> &&
    std::is_same_v<field_type<schema::blocks_esdf, 1>, std::int32_t> &&
    std::is_same_v<field_type<schema::blocks_esdf, 4>, std::uint32_t>);
static_assert(std::is_same_v<field_type<schema::projective_esdf, 0>, float> &&
    std::is_same_v<field_type<schema::projective_esdf, 1>, std::int32_t> &&
    std::is_same_v<field_type<schema::projective_esdf, 4>, std::uint32_t>);
template<schema s> struct aos_tile { std::array<value<s>, voxels_per_tile> voxels{}; };
}
