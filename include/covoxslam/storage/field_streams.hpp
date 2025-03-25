#pragma once
#include <covoxslam/storage/ranges.hpp>

namespace covoxslam::storage
{
enum class stream_order { tile, contiguous };
template<schema s, class accessor> operation_status selective_query(const map_view<s, accessor>& source,
    voxel_range range, std::span<float> output, stream_order order = stream_order::tile);
template<schema s, std::size_t field, class accessor> operation_status gather_field(
    const map_view<s, accessor>& source, voxel_range range, std::span<field_type<s, field>> output);
template<schema s, std::size_t field, class accessor> operation_status assign_field(
    const map_view<s, accessor>& destination, voxel_range range, std::span<const field_type<s, field>> input)
    requires (!accessor::read_only);
template<schema s, std::size_t field, class accessor> operation_status gather_field_indices(
    const map_view<s, accessor>& source, std::span<const std::size_t> indices, std::span<field_type<s, field>> output);
}
