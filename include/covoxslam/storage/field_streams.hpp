#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/ranges.hpp>

namespace covoxslam::storage
{
enum class stream_order { tile, contiguous };
template<schema s, class accessor, class scalar> operation_status validate_stream(
    const map_view<s, accessor>& source, voxel_range range, std::span<scalar> output)
{
    auto checked = validate_range(source, range);
    if (!checked) return checked;
    if (range.count > output.size() || range.count > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(scalar))
        return std::unexpected(operation_error{"field stream extent is invalid"});
    if (overlaps_bytes(source, range, output.data(), range.count * sizeof(scalar)))
        return std::unexpected(operation_error{"field stream aliases collection storage"});
    return {};
}
template<schema s, class accessor> operation_status selective_query(const map_view<s, accessor>& source,
    voxel_range range, std::span<float> output, stream_order order = stream_order::tile)
{
    auto checked = validate_stream(source, range, output);
    if (!checked) return checked;
    if (order != stream_order::tile && order != stream_order::contiguous)
        return std::unexpected(operation_error{"invalid stream traversal order"});
    std::size_t copied{};
    while (copied < range.count)
    {
        const auto first = range.first + copied;
        {
            const auto distances = source.native_accessor().template field_segment<0>(first);
            const auto flags = source.native_accessor().template field_segment<field_count<s> - 1>(first);
            auto length = std::min({distances.size(), flags.size(), range.count - copied});
            if (order == stream_order::tile) length = tile_segment_size(first, length);
            if constexpr (s == schema::tsdf) {
                const auto weights = source.native_accessor().template field_segment<1>(first);
                length = std::min(length, weights.size());

                for (std::size_t i = 0; i < length; ++i)
                    output[copied + i] = (flags[i] & 1U) ? distances[i] * weights[i] : 0;
            }
            else {
                for (std::size_t i = 0; i < length; ++i)
                    output[copied + i] = (flags[i] & 1U) ? distances[i] : 0;
            }
            copied += length;
        }
    }
    return {};
}
template<schema s, std::size_t field, class accessor> operation_status gather_field(
    const map_view<s, accessor>& source, voxel_range range, std::span<field_type<s, field>> output)
{
    auto checked = validate_stream(source, range, output);
    if (!checked) return checked;
    std::size_t copied{};
    while (copied < range.count) {
        const auto first = range.first + copied;
        if constexpr (requires { source.native_accessor().template field_segment<field>(first); }) {
            const auto segment = source.native_accessor().template field_segment<field>(first);
            const auto length = std::min(segment.size(), range.count - copied);
            std::memcpy(output.data() + copied, segment.data(), length * sizeof(field_type<s, field>));
            copied += length;
        }
        else {
            auto v = source[first / 512].voxels[first % 512];
            copy_scalar_bits(output[copied++], std::get<field>(scalar_fields<s>(v)));
        }
    }
    return {};
}
template<schema s, std::size_t field, class accessor> operation_status assign_field(
    const map_view<s, accessor>& destination, voxel_range range, std::span<const field_type<s, field>> input)
    requires (!accessor::read_only)
{
    auto checked = validate_stream(destination, range, input);
    if (!checked) return checked;
    std::size_t copied{};
    while (copied < range.count) {
        const auto first = range.first + copied;
        if constexpr (requires { destination.native_accessor().template field_segment<field>(first); }) {
            const auto segment = destination.native_accessor().template field_segment<field>(first);
            const auto length = std::min(segment.size(), range.count - copied);
            std::memcpy(segment.data(), input.data() + copied, length * sizeof(field_type<s, field>));
            copied += length;
        }
        else {
            auto v = destination[first / 512].voxels[first % 512];
            copy_scalar_bits(std::get<field>(scalar_fields<s>(v)), input[copied++]);
        }
    }
    return {};
}
template<schema s, std::size_t field, class accessor> operation_status gather_field_indices(
    const map_view<s, accessor>& source, std::span<const std::size_t> indices, std::span<field_type<s, field>> output)
{
    auto checked = validate_range(source, {0, 0});
    if (!checked) return checked;
    if (indices.size() > output.size() || indices.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(std::size_t))
        return std::unexpected(operation_error{"indexed field output extent is invalid"});
    const auto bytes = indices.size() * sizeof(field_type<s, field>);
    if (byte_ranges_overlap(indices.data(), indices.size_bytes(), output.data(), bytes) ||
        overlaps_bytes(source, {0, source.size() * 512}, output.data(), bytes))
        return std::unexpected(operation_error{"indexed field output aliases an input"});

    for (auto index : indices) {
        checked = validate_range(source, {index, 1});
        if (!checked) return checked;
    }

    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto v = source[indices[i] / 512].voxels[indices[i] % 512];
        copy_scalar_bits(output[i], std::get<field>(scalar_fields<s>(v)));
    }
    return {};
}
template<schema s, std::size_t field, class accessor> operation_status assign_field_indices(
    const map_view<s, accessor>& destination, std::span<const std::size_t> indices,
    std::span<const field_type<s, field>> input) requires (!accessor::read_only)
{
    auto checked = validate_range(destination, {0, 0});
    if (!checked) return checked;
    if (indices.size() > input.size() || indices.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()) / sizeof(std::size_t))
        return std::unexpected(operation_error{"indexed field input extent is invalid"});
    const voxel_range extent{0, destination.size() * voxels_per_tile};

    if (overlaps_bytes(destination, extent, input.data(), indices.size() * sizeof(field_type<s, field>)) ||
        overlaps_bytes(destination, extent, indices.data(), indices.size_bytes()))
        return std::unexpected(operation_error{"indexed field update aliases its inputs"});

    for (auto index : indices) {
        checked = validate_range(destination, {index, 1});
        if (!checked) return checked;
    }

    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        auto v = destination[indices[i] / 512].voxels[indices[i] % 512];
        copy_scalar_bits(std::get<field>(scalar_fields<s>(v)), input[i]);
    }
    return {};
}
template<schema s, std::size_t field, class accessor, class function>
operation_status update_field(const map_view<s, accessor>& destination, voxel_range range, function&& update)
    requires (!accessor::read_only)
{
    auto checked = validate_range(destination, range);
    if (!checked) return checked;
    std::size_t copied{};
    while (copied < range.count) {
        const auto first = range.first + copied;
        if constexpr (requires { destination.native_accessor().template field_segment<field>(first); }) {
            auto segment = destination.native_accessor().template field_segment<field>(first);
            const auto length = std::min(segment.size(), range.count - copied);

            for (std::size_t i = 0; i < length; ++i) segment[i] = update(segment[i]);
            copied += length;
        }
        else {
            auto v = destination[first / 512].voxels[first % 512];
            auto& scalar = std::get<field>(scalar_fields<s>(v));
            scalar = update(scalar);
            ++copied;
        }
    }
    return {};
}
}
