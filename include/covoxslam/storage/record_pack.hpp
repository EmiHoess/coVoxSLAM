#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/ranges.hpp>

namespace covoxslam::storage
{
struct packing_options { std::size_t batch_records{32}; };
template<std::size_t count> constexpr packing_options fixed_packing()
{
    return {count};
}
inline operation_status validate_packing(const packing_options& options)
{
    if (!options.batch_records || options.batch_records > 4096)
        return std::unexpected(operation_error{"packing batch must be between 1 and 4096 records"});
    return {};
}
template<schema s, class accessor>
auto source_field_segments(const map_view<s, accessor>& source, std::size_t first)
    requires requires { source.native_accessor().template field_segment<0>(first); }
{
    return [&]<std::size_t... field>(std::index_sequence<field...>) {
        return std::tuple{std::span<const field_type<s, field>>(
            source.native_accessor().template field_segment<field>(first))...};
    }(std::make_index_sequence<field_count<s>>{});
}
template<schema s, class accessor>
auto destination_field_segments(const map_view<s, accessor>& destination, std::size_t first)
    requires (!accessor::read_only && requires { destination.native_accessor().template field_segment<0>(first); })
{
    return [&]<std::size_t... field>(std::index_sequence<field...>) {
        return std::tuple{destination.native_accessor().template field_segment<field>(first)...};
    }(std::make_index_sequence<field_count<s>>{});
}
template<class... segment_type> std::size_t common_segment_size(const std::tuple<segment_type...>& segments)
{
    return std::apply([](const auto&... segment) { return std::min({segment.size()...}); }, segments);
}
template<schema s, class accessor> operation_status pack_records(const map_view<s, accessor>& source,
    std::size_t first, std::size_t count, std::span<value<s>> destination, const packing_options& options = {})
{
    auto checked = validate_packing(options);
    if (!checked) return checked;
    checked = validate_range(source, {first, count});
    if (!checked) return checked;
    if (count > destination.size() || count > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(value<s>))
        return std::unexpected(operation_error{"packed destination is too small"});
    if (count == 0) return {};
    {
        if (overlaps_bytes(source, {first, count}, destination.data(), count * sizeof(value<s>)))
            return std::unexpected(operation_error{"packed ranges overlap"});
        std::size_t copied{};
        while (copied < count) {
            const auto segments = source_field_segments(source, first + copied);
            const auto length = std::min(common_segment_size(segments), count - copied);
            std::size_t offset{};
            while (offset < length) {
                const auto batch = std::min(options.batch_records, length - offset);

                for (std::size_t i = 0; i < batch; ++i) {
                    auto fields = scalar_fields<s>(destination[copied + offset + i]);
                    [&]<std::size_t... field>(std::index_sequence<field...>) {
                        (copy_scalar_bits(std::get<field>(fields), std::get<field>(segments)[offset + i]), ...);
                    }(std::make_index_sequence<field_count<s>>{});
                }
                offset += batch;
            }
            copied += length;
        }
        return {};
    }
}
template<schema s, class accessor> operation_status unpack_records(std::span<const value<s>> source,
    const map_view<s, accessor>& destination, std::size_t first, const packing_options& options = {}) requires (!accessor::read_only)
{
    auto checked = validate_packing(options);
    if (!checked) return checked;
    checked = validate_range(destination, {first, source.size()});
    if (!checked) return checked;
    if (source.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(value<s>))
        return std::unexpected(operation_error{"unpacked source size overflows"});
    if (source.empty()) return {};
    if (overlaps_bytes(destination, {first, source.size()}, source.data(), source.size_bytes()))
        return std::unexpected(operation_error{"unpacked ranges overlap"});
    std::size_t copied{};
    while (copied < source.size())
    {
        {
            auto segments = destination_field_segments(destination, first + copied);
            const auto length = std::min(common_segment_size(segments), source.size() - copied);
            std::size_t offset{};
            while (offset < length) {
                const auto batch = std::min(options.batch_records, length - offset);

                for (std::size_t i = 0; i < batch; ++i) {
                    const auto fields = scalar_fields<s>(source[copied + offset + i]);
                    [&]<std::size_t... field>(std::index_sequence<field...>) {
                        (copy_scalar_bits(std::get<field>(segments)[offset + i], std::get<field>(fields)), ...);
                    }(std::make_index_sequence<field_count<s>>{});
                }
                offset += batch;
            }
            copied += length;
        }
    }
    return {};
}
}
