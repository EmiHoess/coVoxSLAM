#pragma once
#include <covoxslam/storage/ranges.hpp>

namespace covoxslam::storage
{
struct packing_options { std::size_t batch_records{32}; };
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
    std::size_t first, std::size_t count, std::span<value<s>> destination, const packing_options& options = {});
template<schema s, class accessor> operation_status unpack_records(std::span<const value<s>> source,
    const map_view<s, accessor>& destination, std::size_t first, const packing_options& options = {}) requires (!accessor::read_only);
}
