#pragma once
#include <covoxslam/storage/draft.hpp>
#include <cstring>

namespace covoxslam::storage
{
struct voxel_range { std::size_t first{}, count{}; };
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
template<schema s, class accessor> bool overlaps_output(const map_view<s, accessor>& source,
    voxel_range range, std::span<value<s>> output)
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
                return byte_ranges_overlap(&begin, bytes, output.data(), output.size_bytes());
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
    if constexpr (requires { source.native_accessor().record_segment(first); }) {
        const auto segment = source.native_accessor().record_segment(first);
        if (count <= segment.size() && segment.data() == destination.data()) return {};
    }
    if (overlaps_output(source, {first, count}, destination.first(count)))
        return std::unexpected(operation_error{"record ranges overlap"});
    if constexpr (requires { source.native_accessor().record_segment(first); })
    {
        static_assert(std::is_trivially_copyable_v<value<s>>);
        std::size_t copied{};
        while (copied < count) {
            const auto segment = source.native_accessor().record_segment(first + copied);
            const auto length = std::min(segment.size(), count - copied);
            std::memcpy(destination.data() + copied, segment.data(), length * sizeof(value<s>));
            copied += length;
        }
    }
    else {
        for (std::size_t i = 0; i < count; ++i)
            destination[i] = gather_record(source[(first + i) / 512].voxels[(first + i) % 512]);
    }
    return {};
}
}
