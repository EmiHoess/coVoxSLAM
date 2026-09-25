#pragma once
#if !defined(__CUDACC__)
#include <covoxslam/core/result.hpp>
#endif
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/schema.hpp>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <tuple>
#include <utility>

#if defined(__CUDACC__)
#define covoxslam_storage_hd __host__ __device__
#else
#define covoxslam_storage_hd
#endif

namespace covoxslam::storage
{
template<schema s, class t> auto scalar_fields(t& record) noexcept
{
    if constexpr (s == schema::tsdf)
        return std::tie(record.distance, 
                        record.weight, 
                        record.sample_count, 
                        record.flags);
    else if constexpr (s == schema::esdf)
        return std::tie(record.signed_distance_m, 
                        record.parent_dx, 
                        record.parent_dy, 
                        record.parent_dz, 
                        record.flags);
    else if constexpr (s == schema::blocks_esdf)
        return std::tie(record.distance_m, 
                        record.parent_dx, 
                        record.parent_dy, 
                        record.parent_dz, 
                        record.flags);
    else return std::tie(record.squared_distance_voxels, 
                        record.parent_x, 
                        record.parent_y, 
                        record.parent_z, 
                        record.flags);
}
template<bool read_only, class t> using scalar_reference = std::conditional_t<read_only, const t&, t&>;
template<schema s, bool read_only> struct voxel_reference;
template<bool read_only> struct voxel_reference<schema::tsdf, read_only>
{
    scalar_reference<read_only, float> distance, weight;
    scalar_reference<read_only, std::uint32_t> sample_count, flags;
};
template<bool read_only> struct voxel_reference<schema::esdf, read_only> {
    scalar_reference<read_only, float> signed_distance_m;
    scalar_reference<read_only, std::int32_t> parent_dx, parent_dy, parent_dz;
    scalar_reference<read_only, std::uint32_t> flags;
};
template<bool read_only> struct voxel_reference<schema::blocks_esdf, read_only>
{
    scalar_reference<read_only, float> distance_m;
    scalar_reference<read_only, std::int32_t> parent_dx, parent_dy, parent_dz;
    scalar_reference<read_only, std::uint32_t> flags;
};
template<bool read_only> struct voxel_reference<schema::projective_esdf, read_only>
{
    scalar_reference<read_only, float> squared_distance_voxels;
    scalar_reference<read_only, std::int32_t> parent_x, parent_y, parent_z;
     scalar_reference<read_only, std::uint32_t> flags;
};
template<schema s, class accessor>
covoxslam_storage_hd auto bind_voxel(const accessor& access, std::size_t tile, std::size_t voxel) noexcept
{
    if constexpr (s == schema::tsdf)
        return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
            access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel),
            access.template scalar<3>(tile, voxel)};
    else
        return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
            access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel),
            access.template scalar<3>(tile, voxel), access.template scalar<4>(tile, voxel)};
}
template<schema s, class accessor> struct voxel_array_view
{
    accessor access;
    std::size_t tile{};
    covoxslam_storage_hd constexpr std::size_t size() const noexcept { return tile_voxels; }
    covoxslam_storage_hd auto operator[](std::size_t voxel) const noexcept
    {
        return bind_voxel<s>(access, tile, voxel);
    }
};
template<schema s, class accessor> struct tile_view { voxel_array_view<s, accessor> voxels; };
template<class accessor> struct readonly_accessor
{
    accessor source;
    static constexpr bool read_only = true;
    covoxslam_storage_hd std::size_t size() const noexcept { return source.size(); }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return source.capacity(); }
    covoxslam_storage_hd bool valid() const noexcept { return source.valid(); }
    const void* owner_identity() const noexcept requires requires { source.owner_identity(); }
    {
        return source.owner_identity();
    }
    auto record_segment(std::size_t first) const noexcept requires requires { source.record_segment(first); }
    {
        const auto segment = source.record_segment(first);
        return std::span<const typename decltype(segment)::element_type>(segment);
    }
    template<std::size_t field> auto field_segment(std::size_t first) const noexcept
        requires requires { source.template field_segment<field>(first); }
    {
        const auto segment = source.template field_segment<field>(first);
        return std::span<const typename decltype(segment)::element_type>(segment);
    }
    template<std::size_t field>
    covoxslam_storage_hd const auto& scalar(std::size_t tile, std::size_t voxel) const noexcept
    {
        return source.template scalar<field>(tile, voxel);
    }
};
template<schema s, class accessor> class map_view
{
    accessor access_;
public:
    covoxslam_storage_hd explicit map_view(accessor access) noexcept : access_(access) {}
    covoxslam_storage_hd std::size_t size() const noexcept { return access_.size(); }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return access_.capacity(); }
    covoxslam_storage_hd bool empty() const noexcept { return size() == 0; }
    covoxslam_storage_hd bool valid() const noexcept { return access_.valid(); }
    const accessor& native_accessor() const noexcept { return access_; }
    covoxslam_storage_hd auto as_const() const noexcept
    {
        return map_view<s, readonly_accessor<accessor>>{readonly_accessor<accessor>{access_}};
    }
    covoxslam_storage_hd auto operator[](std::size_t tile) const noexcept
    {
        return tile_view<s, accessor>{{access_, tile}};
    }
};
template<schema s, bool constant>
covoxslam_storage_hd value<s> gather_record(const voxel_reference<s, constant>& record) noexcept
{
    if constexpr (s == schema::tsdf)
        return {record.distance, record.weight, record.sample_count, record.flags};
    else if constexpr (s == schema::esdf)
        return {record.signed_distance_m, record.parent_dx, record.parent_dy, record.parent_dz, record.flags};
    else if constexpr (s == schema::blocks_esdf)
        return {record.distance_m, record.parent_dx, record.parent_dy, record.parent_dz, record.flags};
    else return {record.squared_distance_voxels, record.parent_x, record.parent_y, record.parent_z, record.flags};
}
template<schema s>
covoxslam_storage_hd void scatter_record(voxel_reference<s, false> target, value<s> source) noexcept
{
    if constexpr (s == schema::tsdf)
    {
        target.distance = source.distance;
        target.weight = source.weight;
        target.sample_count = source.sample_count;
    }
    else if constexpr (s == schema::projective_esdf) {
        target.squared_distance_voxels = source.squared_distance_voxels;
        target.parent_x = source.parent_x;
        target.parent_y = source.parent_y;
        target.parent_z = source.parent_z;
    }
    else
    {
        if constexpr (s == schema::esdf) target.signed_distance_m = source.signed_distance_m;
        else target.distance_m = source.distance_m;
        target.parent_dx = source.parent_dx;
        target.parent_dy = source.parent_dy;
        target.parent_dz = source.parent_dz;
    }
     target.flags = source.flags;
}
template<schema s, class accessor> struct device_voxel_array
{
    accessor access;
    std::size_t tile{};
    covoxslam_storage_hd constexpr std::size_t size() const noexcept { return tile_voxels; }
#if defined(__CUDACC__)
    __device__ auto operator[](std::size_t voxel) const noexcept
    {
        if constexpr (s == schema::tsdf)
            return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
                access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel), access.template scalar<3>(tile, voxel)};
        else
            return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
                access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel),
                access.template scalar<3>(tile, voxel), access.template scalar<4>(tile, voxel)};
    }
#endif
};
template<schema s, class accessor> struct device_tile_view { device_voxel_array<s, accessor> voxels; };
template<schema s, class accessor> class device_collection
{
    accessor access_;
public:
    covoxslam_storage_hd explicit device_collection(accessor access) noexcept : access_(access) {}
    covoxslam_storage_hd std::size_t size() const noexcept { return access_.size(); }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return access_.capacity(); }
    covoxslam_storage_hd bool empty() const noexcept { return size() == 0; }
    covoxslam_storage_hd bool valid() const noexcept { return access_.valid(); }
    covoxslam_storage_hd const accessor& native_accessor() const noexcept { return access_; }
#if defined(__CUDACC__)
    __device__ auto operator[](std::size_t tile) const noexcept
    {
        return device_tile_view<s, accessor>{{access_, tile}};
    }
#endif
};

struct device_descriptor
{
    std::byte* fields[maximum_schema_fields]{};
    std::size_t size_tiles{}, capacity_tiles{}, tile_stride{}, voxel_stride{};
    std::uint64_t generation{};
    schema record_schema{schema::tsdf};
    layout storage_layout{layout::pool_soa};
    int device{-1};
};
covoxslam_storage_hd constexpr bool valid_pool_soa_descriptor_shape(
    const device_descriptor& descriptor) noexcept
{
    return descriptor.storage_layout == layout::pool_soa &&
        descriptor.tile_stride == pool_soa_tile_stride_bytes &&
        descriptor.voxel_stride == pool_soa_voxel_stride_bytes &&
        descriptor.size_tiles <= descriptor.capacity_tiles;
}
template<schema s, bool constant = false> struct device_accessor
{
    device_descriptor descriptor;
    static constexpr bool read_only = constant;
    covoxslam_storage_hd std::size_t size() const noexcept { return descriptor.size_tiles; }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return descriptor.capacity_tiles; }
    covoxslam_storage_hd bool valid() const noexcept
    {
        return descriptor.record_schema == s && valid_pool_soa_descriptor_shape(descriptor) &&
            descriptor.generation != 0;
    }
#if defined(__CUDACC__)
    template<std::size_t field>
    __device__ scalar_reference<constant, field_type<s, field>> scalar(std::size_t tile, std::size_t voxel) const noexcept
    {
        using scalar_type = std::conditional_t<constant, const field_type<s, field>, field_type<s, field>>;
        return *reinterpret_cast<scalar_type*>(descriptor.fields[field] + tile * descriptor.tile_stride + voxel * descriptor.voxel_stride);
    }
#endif
};

template<schema s, bool constant = false> using device_map_view = device_collection<s, device_accessor<s, constant>>;
int initialize_device_fields(device_descriptor descriptor, void* stream) noexcept;

#if !defined(__CUDACC__)

template<schema s, class accessor>
operation_result<tile_view<s, accessor>> checked_at(const map_view<s, accessor>& view, std::size_t tile)
{
    if (!view.valid()) return std::unexpected(operation_error{"storage view is stale"});
    
    if (tile >= view.size()) return std::unexpected(operation_error{"tile index is outside collection"});
    return view[tile];
}
template<schema s, class accessor>
operation_result<voxel_reference<s, accessor::read_only>> checked_at(
    const map_view<s, accessor>& view, std::size_t tile, std::size_t voxel)
{
    auto selected = checked_at(view, tile);

    if (!selected) return std::unexpected(selected.error());
    
    if (voxel >= tile_voxels) return std::unexpected(operation_error{"voxel index is outside tile"});
    return selected->voxels[voxel];
}
struct voxel_range { std::size_t first{}, count{}; };
template<class t> void copy_scalar_bits(t& destination, const t& source) noexcept
{
    std::memcpy(&destination, &source, sizeof(t));
}
template<schema s, class accessor> operation_status validate_range(const map_view<s, accessor>& source, voxel_range range)
{
    if (!source.valid()) return std::unexpected(operation_error{"range source view is stale"});

    if (source.size() > std::numeric_limits<std::size_t>::max() / tile_voxels)
        return std::unexpected(operation_error{"range extent overflows"});
    
        const auto extent = source.size() * tile_voxels;
    if (range.first > extent || range.count > extent - range.first)
        return std::unexpected(operation_error{"voxel range exceeds collection"});
    
        return {};
}
inline std::size_t tile_segment_size(std::size_t first, std::size_t remaining) noexcept
{
    return std::min(remaining, tile_voxels - first % tile_voxels);
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
        auto first = source[range.first / tile_voxels].voxels[range.first % tile_voxels];
        auto last = source[range.first / tile_voxels].voxels[range.first % tile_voxels + count - 1];
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
            auto input = source[(first + i) / tile_voxels].voxels[(first + i) % tile_voxels];
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
            auto input = source[a / tile_voxels].voxels[a % tile_voxels];
            auto output = destination[b / tile_voxels].voxels[b % tile_voxels];
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
    std::uint32_t selected_fields = all_field_mask<s>) requires (!destination_accessor::read_only)
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
#endif
}
