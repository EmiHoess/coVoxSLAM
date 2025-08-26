#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/host_buffer.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/ranges.hpp>
#include <covoxslam/storage/device_view.hpp>

namespace covoxslam::storage
{
enum class transfer_method { fieldwise_reference, bulk_same_layout };
struct transfer_endpoint
{
    std::byte* fields[5]{};
    std::size_t tiles{}, tile_stride{}, voxel_stride{};
    schema record_schema{schema::tsdf};
    layout storage_layout{layout::pool_soa};
    int device{-1};
};
struct transfer_region {
    const void* source{};
    void* destination{};
    std::size_t source_pitch{}, destination_pitch{}, width{}, height{};
};
struct transfer_plan
{
    std::vector<transfer_region> regions;
    std::size_t useful_bytes{};
    bool fallback{};
};
inline transfer_endpoint transfer_storage(const device_descriptor& source)
{
    transfer_endpoint result;
    std::copy_n(source.fields, 5, result.fields);
    result.tiles = source.size_tiles;
    result.tile_stride = source.tile_stride;
    result.voxel_stride = source.voxel_stride;
    result.record_schema = source.record_schema;
    result.storage_layout = source.storage_layout;
    result.device = source.device;
    return result;
}
template<schema s, layout l> transfer_endpoint transfer_storage(const host_buffer<s, l>& source)
{
    transfer_endpoint result;
    result.tiles = source.size();
    result.record_schema = s;
    result.storage_layout = l;
    result.tile_stride = storage_traits<s, l>::tile_stride;
    result.voxel_stride = storage_traits<s, l>::voxel_stride;

    if (source.size()) [&]<std::size_t... field>(std::index_sequence<field...>) {
        ((result.fields[field] = reinterpret_cast<std::byte*>(const_cast<field_type<s, field>*>(&source.template scalar<field>(0, 0)))), ...);
    }(std::make_index_sequence<field_count<s>>{});
    return result;
}
inline std::size_t transfer_fields(schema s)
{
    switch (s) {
    case schema::tsdf: return 4;
    case schema::esdf: case schema::blocks_esdf: case schema::projective_esdf: return 5;
    default: return 0;
    }
}
inline operation_status validate_endpoint(const transfer_endpoint& value)
{
    constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    const auto fields = transfer_fields(value.record_schema);
    if (!fields || value.device < -1 || value.tiles > maximum / 512 || !value.tile_stride ||
        value.tile_stride > maximum / std::max<std::size_t>(value.tiles, 1))
        return std::unexpected(operation_error{"invalid transfer descriptor extent"});
    if (value.storage_layout != layout::pool_soa)
        return std::unexpected(operation_error{"transfer requires SoA storage"});
    if (value.voxel_stride != 4 || value.tile_stride != 2048)
        return std::unexpected(operation_error{"invalid SoA transfer strides"});
    const auto bytes = value.tiles * value.tile_stride;

    for (std::size_t field = 0; field < fields; ++field) {
        const auto address = reinterpret_cast<std::uintptr_t>(value.fields[field]);
        if (value.tiles && (!address || address % 4 || address > std::numeric_limits<std::uintptr_t>::max() - bytes))
            return std::unexpected(operation_error{"invalid transfer field address"});
    }
    if (value.tiles) {

        for (std::size_t a = 0; a < fields; ++a) {

            for (std::size_t b = a + 1; b < fields; ++b) {
                if (byte_ranges_overlap(value.fields[a], bytes, value.fields[b], bytes))
                    return std::unexpected(operation_error{"transfer fields overlap"});
            }
        }
    }
    return {};
}
inline std::byte* transfer_address(const transfer_endpoint& value, std::size_t field, std::size_t voxel)
{
    return reinterpret_cast<std::byte*>(reinterpret_cast<std::uintptr_t>(value.fields[field]) +
        voxel / 512 * value.tile_stride + voxel % 512 * value.voxel_stride);
}
inline operation_status append_transfer(transfer_plan& plan, transfer_region region, std::size_t maximum_pitch)
{
    constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    if (!region.height || !region.width) return {};
    if (region.source_pitch < region.width || region.destination_pitch < region.width ||
        (region.height > 1 && (region.source_pitch > maximum_pitch || region.destination_pitch > maximum_pitch)) || region.width > maximum ||
        region.height - 1 > (maximum - region.width) / region.source_pitch ||
        region.height - 1 > (maximum - region.width) / region.destination_pitch ||
        region.height > (maximum - plan.useful_bytes) / region.width)
        return std::unexpected(operation_error{"transfer pitch or extent exceeds limits"});
    const auto source_bytes = (region.height - 1) * region.source_pitch + region.width;
    const auto destination_bytes = (region.height - 1) * region.destination_pitch + region.width;
    if (!region.source || !region.destination ||
        std::uintptr_t(region.source) > std::numeric_limits<std::uintptr_t>::max() - source_bytes ||
        std::uintptr_t(region.destination) > std::numeric_limits<std::uintptr_t>::max() - destination_bytes)
        return std::unexpected(operation_error{"transfer address extent overflows"});
    plan.useful_bytes += region.width * region.height;
    plan.regions.push_back(region);
    return {};
}
inline operation_result<transfer_plan> plan_transfer(const transfer_endpoint& source, std::size_t source_first,
    const transfer_endpoint& destination, std::size_t destination_first, std::size_t count,
    std::uint32_t fields, transfer_method method, std::size_t maximum_pitch)
{
    auto checked = validate_endpoint(source);
    if (!checked) return std::unexpected(checked.error());
    checked = validate_endpoint(destination);
    if (!checked) return std::unexpected(checked.error());
    const auto field_count = transfer_fields(source.record_schema);
    if (source.record_schema != destination.record_schema || fields >> field_count ||
        source_first > source.tiles * 512 || count > source.tiles * 512 - source_first ||
        destination_first > destination.tiles * 512 || count > destination.tiles * 512 - destination_first ||
        (source.device >= 0 && destination.device >= 0 && source.device != destination.device) ||
        (method != transfer_method::fieldwise_reference && method != transfer_method::bulk_same_layout))
        return std::unexpected(operation_error{"incompatible transfer ranges"});
    transfer_plan result;
    if (!count || !fields) return result;
    bool identical = source.device == destination.device && source_first == destination_first &&
        source.tile_stride == destination.tile_stride && source.voxel_stride == destination.voxel_stride;

    for (std::size_t f = 0; f < field_count; ++f)
        if (fields & (1U << f)) identical = identical && source.fields[f] == destination.fields[f];
    if (identical) return result;
    if (source.device == destination.device) {
        for (std::size_t a = 0; a < field_count; ++a)
            for (std::size_t b = 0; b < field_count; ++b) {
                if (!(fields & (1U << a)) || !(fields & (1U << b))) continue;
                const auto* begin = transfer_address(source, a, source_first);
                const auto* end = transfer_address(source, a, source_first + count - 1);
                const auto* target = transfer_address(destination, b, destination_first);
                const auto* last = transfer_address(destination, b, destination_first + count - 1);
                if (byte_ranges_overlap(begin, std::uintptr_t(end) - std::uintptr_t(begin) + 4,
                    target, std::uintptr_t(last) - std::uintptr_t(target) + 4))
                    return std::unexpected(operation_error{"transfer payloads overlap"});
            }
    }
    const bool matching = source.storage_layout == destination.storage_layout;
    const bool bulk = method == transfer_method::bulk_same_layout && matching;
    result.fallback = method == transfer_method::bulk_same_layout && !matching;
    std::size_t copied{};
    while (copied < count) {
        const auto a = source_first + copied, b = destination_first + copied;
        if (bulk && source.storage_layout == layout::pool_soa && source.tile_stride == 2048 && destination.tile_stride == 2048) {
            const auto bytes = (count - copied) * 4;

            for (std::size_t f = 0; f < field_count; ++f) {
                if (!(fields & (1U << f))) continue;
                checked = append_transfer(result, {transfer_address(source, f, a), transfer_address(destination, f, b),
                    bytes, bytes, bytes, 1}, maximum_pitch);
                if (!checked) return std::unexpected(checked.error());
            }
            copied = count;
            continue;
        }
        const auto length = std::min({count - copied, 512 - a % 512, 512 - b % 512});

        for (std::size_t field = 0; field < field_count; ++field) {
            if (!(fields & (1U << field))) continue;
            if (method == transfer_method::bulk_same_layout && source.voxel_stride == 4 && destination.voxel_stride == 4) {
                checked = append_transfer(result, {transfer_address(source, field, a), transfer_address(destination, field, b),
                    length * 4, length * 4, length * 4, 1}, maximum_pitch);
                result.fallback = true;
            }
            else {
                checked = append_transfer(result, {transfer_address(source, field, a), transfer_address(destination, field, b),
                    source.voxel_stride, destination.voxel_stride, 4, length}, maximum_pitch);
            }
            if (!checked) return std::unexpected(checked.error());
        }
        copied += length;
    }
    return result;
}
}
