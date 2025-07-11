#pragma once
#include <covoxslam/storage/draft.hpp>
#include <cstring>

namespace covoxslam::storage
{
struct voxel_range { std::size_t first{}, count{}; };
template<class t> void copy_scalar_bits(t& destination, const t& source) noexcept
{
    std::memcpy(&destination, &source, sizeof(t));
}
template<schema s, class accessor> operation_status validate_range(const map_view<s, accessor>& source, voxel_range range)
{
    if (!source.valid()) return std::unexpected(operation_error{"range source view is stale"});
    if (source.size() > std::numeric_limits<std::size_t>::max() / voxels_per_tile)
        return std::unexpected(operation_error{"range extent overflows"});
    const auto extent = source.size() * voxels_per_tile;
    if (range.first > extent || range.count > extent - range.first)
        return std::unexpected(operation_error{"voxel range exceeds collection"});
    return {};
}
inline std::size_t tile_segment_size(std::size_t first, std::size_t remaining) noexcept
{
    return std::min(remaining, voxels_per_tile - first % voxels_per_tile);
}
inline bool byte_ranges_overlap(const void* first, std::size_t first_bytes, const void* second, std::size_t second_bytes) noexcept
{
    if (!first_bytes || !second_bytes) return false;
    const auto a = reinterpret_cast<std::uintptr_t>(first), b = reinterpret_cast<std::uintptr_t>(second);
    return a <= b ? b - a < first_bytes : a - b < second_bytes;
}
template<schema s, class accessor> bool overlaps_bytes(const map_view<s, accessor>& source,
    voxel_range range, const void* output, std::size_t output_bytes)
{
    while (range.count)
    {
        const auto count = tile_segment_size(range.first, range.count);
        auto first = source[range.first / 512].voxels[range.first % 512];
        auto last = source[range.first / 512].voxels[range.first % 512 + count - 1];
        const auto begins = scalar_fields<s>(first), ends = scalar_fields<s>(last);
        const bool overlap = [&]<std::size_t... field>(std::index_sequence<field...>) {
            const auto check = [&](const auto& begin, const auto& end) {
                const auto bytes = reinterpret_cast<std::uintptr_t>(&end) - reinterpret_cast<std::uintptr_t>(&begin) + sizeof(begin);
                return byte_ranges_overlap(&begin, bytes, output, output_bytes);
            };
            return (check(std::get<field>(begins), std::get<field>(ends)) || ...);
        }(std::make_index_sequence<field_count<s>>{});

        if (overlap) return true;
        range.first += count;
        range.count -= count;
    }
    return false;
}
template<schema s, class accessor> operation_status copy_records(const map_view<s, accessor>& source,
    std::size_t first, std::size_t count, std::span<value<s>> destination)
{
    auto checked = validate_range(source, {first, count});
    if (!checked) return checked;
    if (count > destination.size() || count > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(value<s>))
        return std::unexpected(operation_error{"record destination is too small"});
    if (count == 0) return {};
    if (overlaps_bytes(source, {first, count}, destination.data(), count * sizeof(value<s>)))
        return std::unexpected(operation_error{"record ranges overlap"});
    {
        for (std::size_t i = 0; i < count; ++i) {
            auto input = source[(first + i) / 512].voxels[(first + i) % 512];
            const auto fields = scalar_fields<s>(input);
            auto output = scalar_fields<s>(destination[i]);
            [&]<std::size_t... field>(std::index_sequence<field...>) {
                (copy_scalar_bits(std::get<field>(output), std::get<field>(fields)), ...);
            }(std::make_index_sequence<field_count<s>>{});
        }
    }
    return {};
}
template<schema s, std::size_t field, class source_accessor, class destination_accessor>
void copy_field_range(const map_view<s, source_accessor>& source, std::size_t source_first,
    const map_view<s, destination_accessor>& destination, std::size_t destination_first, std::size_t count)
{
    std::size_t copied{};
    while (copied < count)
    {
        if constexpr (requires { source.native_accessor().template field_segment<field>(source_first);
                                 destination.native_accessor().template field_segment<field>(destination_first); }) {
            const auto input = source.native_accessor().template field_segment<field>(source_first + copied);
            auto output = destination.native_accessor().template field_segment<field>(destination_first + copied);
            const auto length = std::min({input.size(), output.size(), count - copied});
            std::memcpy(output.data(), input.data(), length * sizeof(field_type<s, field>));
            copied += length;
        }
        else {
            const auto a = source_first + copied, b = destination_first + copied;
            auto input = source[a / 512].voxels[a % 512];
            auto output = destination[b / 512].voxels[b % 512];
            auto&& value = std::get<field>(scalar_fields<s>(input));
            auto&& target = std::get<field>(scalar_fields<s>(output));
            copy_scalar_bits(target, value);
            ++copied;
        }
    }
}
template<schema s, class source_accessor, class destination_accessor>
operation_status copy_fields(const map_view<s, source_accessor>& source, std::size_t source_first,
    const map_view<s, destination_accessor>& destination, std::size_t destination_first, std::size_t count,
    std::uint32_t selected_fields = (1U << field_count<s>) - 1) requires (!destination_accessor::read_only)
{
    auto checked = validate_range(source, {source_first, count});
    if (!checked) return checked;
    checked = validate_range(destination, {destination_first, count});
    if (!checked) return checked;
    if (selected_fields >> field_count<s>) return std::unexpected(operation_error{"unsupported field selection"});
    if (count == 0 || selected_fields == 0) return {};
    if (source.native_accessor().owner_identity() == destination.native_accessor().owner_identity()) {
        if (source_first == destination_first) return {};
        if (source_first < destination_first ? destination_first - source_first < count : source_first - destination_first < count)
            return std::unexpected(operation_error{"field ranges overlap"});
    }
    [&]<std::size_t... field>(std::index_sequence<field...>) {
        ((selected_fields & (1U << field) ? copy_field_range<s, field>(source, source_first, destination, destination_first, count) : void()), ...);
    }(std::make_index_sequence<field_count<s>>{});
    return {};
}
}
