#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/host_buffer.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/views.hpp>
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <span>
#include <tuple>
#include <utility>
#include <vector>
#if defined(covoxslam_has_cuda)
#include <cuda_runtime_api.h>
#include <chrono>
#include <memory>
#endif

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
            auto v = source[first / tile_voxels].voxels[first % tile_voxels];
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
            auto v = destination[first / tile_voxels].voxels[first % tile_voxels];
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
        overlaps_bytes(source, {0, source.size() * tile_voxels}, output.data(), bytes))
        return std::unexpected(operation_error{"indexed field output aliases an input"});

    for (auto index : indices) {
        checked = validate_range(source, {index, 1});
        if (!checked) return checked;
    }

    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto v = source[indices[i] / tile_voxels].voxels[indices[i] % tile_voxels];
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
    const voxel_range extent{0, destination.size() * tile_voxels};

    if (overlaps_bytes(destination, extent, input.data(), indices.size() * sizeof(field_type<s, field>)) ||
        overlaps_bytes(destination, extent, indices.data(), indices.size_bytes()))
        return std::unexpected(operation_error{"indexed field update aliases its inputs"});

    for (auto index : indices) {
        checked = validate_range(destination, {index, 1});
        if (!checked) return checked;
    }

    for (std::size_t i = 0; i < indices.size(); ++i)
    {
        auto v = destination[indices[i] / tile_voxels].voxels[indices[i] % tile_voxels];
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
            auto v = destination[first / tile_voxels].voxels[first % tile_voxels];
            auto& scalar = std::get<field>(scalar_fields<s>(v));
            scalar = update(scalar);
            ++copied;
        }
    }
    return {};
}
enum class transfer_method { fieldwise_reference, bulk_same_layout };
struct transfer_endpoint
{
    std::byte* fields[maximum_schema_fields]{};
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
    std::copy_n(source.fields, maximum_schema_fields, result.fields);
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
inline operation_status validate_endpoint(const transfer_endpoint& value)
{
    constexpr auto maximum = std::size_t(std::numeric_limits<std::ptrdiff_t>::max());
    const auto fields = schema_field_count(value.record_schema);
    if (!fields || value.device < -1 || value.tiles > maximum / tile_voxels || !value.tile_stride ||
        value.tile_stride > maximum / std::max<std::size_t>(value.tiles, 1))
        return std::unexpected(operation_error{"invalid transfer descriptor extent"});
    if (value.storage_layout != layout::pool_soa)
        return std::unexpected(operation_error{"transfer requires SoA storage"});
    if (value.voxel_stride != pool_soa_voxel_stride_bytes ||
        value.tile_stride != pool_soa_tile_stride_bytes)
        return std::unexpected(operation_error{"invalid SoA transfer strides"});
    const auto bytes = value.tiles * value.tile_stride;

    for (std::size_t field = 0; field < fields; ++field) {
        const auto address = reinterpret_cast<std::uintptr_t>(value.fields[field]);
        if (value.tiles && (!address || address % pool_soa_scalar_alignment ||
            address > std::numeric_limits<std::uintptr_t>::max() - bytes))
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
        voxel / tile_voxels * value.tile_stride + voxel % tile_voxels * value.voxel_stride);
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
    const auto field_count = schema_field_count(source.record_schema);
    if (source.record_schema != destination.record_schema || fields >> field_count ||
        source_first > source.tiles * tile_voxels || count > source.tiles * tile_voxels - source_first ||
        destination_first > destination.tiles * tile_voxels || count > destination.tiles * tile_voxels - destination_first ||
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
                if (byte_ranges_overlap(begin,
                    std::uintptr_t(end) - std::uintptr_t(begin) + pool_soa_voxel_stride_bytes,
                    target, std::uintptr_t(last) - std::uintptr_t(target) + pool_soa_voxel_stride_bytes))
                    return std::unexpected(operation_error{"transfer payloads overlap"});
            }
    }
    const bool matching = source.storage_layout == destination.storage_layout;
    const bool bulk = method == transfer_method::bulk_same_layout && matching;
    result.fallback = method == transfer_method::bulk_same_layout && !matching;
    std::size_t copied{};
    while (copied < count) {
        const auto a = source_first + copied, b = destination_first + copied;
        if (bulk && source.storage_layout == layout::pool_soa &&
            source.tile_stride == pool_soa_tile_stride_bytes &&
            destination.tile_stride == pool_soa_tile_stride_bytes) {
            const auto bytes = (count - copied) * pool_soa_voxel_stride_bytes;

            for (std::size_t f = 0; f < field_count; ++f) {
                if (!(fields & (1U << f))) continue;
                checked = append_transfer(result, {transfer_address(source, f, a), transfer_address(destination, f, b),
                    bytes, bytes, bytes, 1}, maximum_pitch);
                if (!checked) return std::unexpected(checked.error());
            }
            copied = count;
            continue;
        }
        const auto length = std::min({count - copied, tile_voxels - a % tile_voxels,
            tile_voxels - b % tile_voxels});

        for (std::size_t field = 0; field < field_count; ++field) {
            if (!(fields & (1U << field))) continue;
            if (method == transfer_method::bulk_same_layout &&
                source.voxel_stride == pool_soa_voxel_stride_bytes &&
                destination.voxel_stride == pool_soa_voxel_stride_bytes) {
                checked = append_transfer(result, {transfer_address(source, field, a), transfer_address(destination, field, b),
                    length * pool_soa_voxel_stride_bytes, length * pool_soa_voxel_stride_bytes,
                    length * pool_soa_voxel_stride_bytes, 1}, maximum_pitch);
                result.fallback = true;
            }
            else {
                checked = append_transfer(result, {transfer_address(source, field, a), transfer_address(destination, field, b),
                    source.voxel_stride, destination.voxel_stride,
                    pool_soa_voxel_stride_bytes, length}, maximum_pitch);
            }
            if (!checked) return std::unexpected(checked.error());
        }
        copied += length;
    }
    return result;
}
#if defined(covoxslam_has_cuda)
struct transfer_measurement
{
    std::size_t calls{}, useful_bytes{}, waits{};
    double planning_ms{}, submission_ms{}, completion_ms{};
    bool fallback{};
};
inline operation_result<transfer_plan> plan_cuda_transfer(const transfer_endpoint& source, std::size_t source_first,
    const transfer_endpoint& destination, std::size_t destination_first, std::size_t count, std::uint32_t fields,
    transfer_method method, transfer_measurement& measurement)
{
    const auto start = std::chrono::steady_clock::now();
    int device{-1}, pitch{};
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaDeviceGetAttribute(&pitch, cudaDevAttrMaxPitch, device);
    if (status != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(status)});
    if ((source.device >= 0 && source.device != device) || (destination.device >= 0 && destination.device != device))
        return std::unexpected(operation_error{"transfer requires the allocation device"});
    auto result = plan_transfer(source, source_first, destination, destination_first, count, fields, method, std::size_t(pitch));
    measurement.planning_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (result) { measurement.useful_bytes = result->useful_bytes; measurement.fallback = result->fallback; }
    return result;
}
inline operation_status execute_cuda_transfer(const transfer_plan& plan, cudaMemcpyKind kind, cudaStream_t stream,
    transfer_measurement& measurement)
{
    const auto start = std::chrono::steady_clock::now();
    cudaError_t submitted = cudaSuccess;

    for (const auto& region : plan.regions)
    {
        ++measurement.calls;
        submitted = region.height == 1 ? cudaMemcpyAsync(region.destination, region.source, region.width, kind, stream) :
            cudaMemcpy2DAsync(region.destination, region.destination_pitch, region.source, region.source_pitch,
                region.width, region.height, kind, stream);
        if (submitted != cudaSuccess) break;
    }
    const auto boundary = std::chrono::steady_clock::now();
    measurement.submission_ms = std::chrono::duration<double, std::milli>(boundary - start).count();
    const auto completed = cudaStreamSynchronize(stream);
    ++measurement.waits;
    measurement.completion_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - boundary).count();
    if (submitted != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(submitted)});
    if (completed != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(completed)});
    return {};
}
struct completion_state
{
    std::shared_ptr<void> retained;
    cudaEvent_t event{};
    cudaStream_t stream{};
    int device{-1};
    bool recorded{};
    ~completion_state()
    {
        int previous{-1};
        const auto queried = cudaGetDevice(&previous);
        if (device < 0) { if (stream) cudaStreamSynchronize(stream); return; }
        if (cudaSetDevice(device) != cudaSuccess) return;
        if (recorded) cudaEventSynchronize(event);
        else if (stream) cudaStreamSynchronize(stream);
        if (event) cudaEventDestroy(event);
        if (queried == cudaSuccess && previous != device) cudaSetDevice(previous);
    }
};
class completion_ticket
{
    std::shared_ptr<completion_state> state_;
    explicit completion_ticket(std::shared_ptr<completion_state> state) : state_(std::move(state)) {}
public:
    completion_ticket() = default;
    completion_ticket(const completion_ticket&) = delete;
    completion_ticket& operator=(const completion_ticket&) = delete;
    completion_ticket(completion_ticket&&) noexcept = default;
    completion_ticket& operator=(completion_ticket&&) noexcept = default;
    static operation_result<completion_ticket> record(cudaStream_t stream, std::shared_ptr<void> retained)
    {
        auto state = std::make_shared<completion_state>();
        state->stream = stream;
        state->retained = std::move(retained);
        auto error = cudaGetDevice(&state->device);
        if (error == cudaSuccess) error = cudaEventCreateWithFlags(&state->event, cudaEventDisableTiming);
        if (error == cudaSuccess) error = cudaEventRecord(state->event, stream);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        state->recorded = true;
        return completion_ticket(std::move(state));
    }
    operation_result<bool> query() const
    {
        if (!state_) return std::unexpected(operation_error{"completion ticket is empty"});
        int current{-1};
        auto error = cudaGetDevice(&current);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        if (current != state_->device) return std::unexpected(operation_error{"completion requires the allocation device"});
        error = cudaEventQuery(state_->event);
        if (error == cudaErrorNotReady) return false;
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        return true;
    }
    operation_status wait() const
    {
        auto available = query();
        if (!available) return std::unexpected(available.error());
        const auto error = *available ? cudaSuccess : cudaEventSynchronize(state_->event);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        return {};
    }
    operation_status depend_on(cudaStream_t stream) const
    {
        auto available = query();
        if (!available) return std::unexpected(available.error());
        const auto error = cudaStreamWaitEvent(stream, state_->event, 0);
        if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
        return {};
    }
    operation_status cancel()
    {
        auto checked = wait();
        if (!checked) return checked;
        state_.reset();
        return {};
    }
};
struct stream_drain
{
    cudaStream_t stream{};
    int device{-1};
    bool armed{true};
    ~stream_drain()
    {
        if (!armed) return;
        int previous{-1};
        const auto queried = cudaGetDevice(&previous);
        if (device < 0 || cudaSetDevice(device) != cudaSuccess) return;
        cudaStreamSynchronize(stream);
        if (queried == cudaSuccess && previous != device) cudaSetDevice(previous);
    }
};
inline operation_status cuda_storage_status(cudaError_t status)
{
    if (status != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(status)});
    return {};
}
#endif
}
