#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/store/store.hpp>
#include <bit>

namespace covoxslam::store
{
struct checkpoint_parameters
{
    double voxel_size{}, truncation{}, maximum_weight{}, esdf_distance{};
    std::uint64_t algorithm{};
    bool operator==(const checkpoint_parameters&) const = default;
};
inline bool valid_parameters(const checkpoint_parameters& value)
{
    return std::isfinite(value.voxel_size) && value.voxel_size > 0 &&
        std::isfinite(value.truncation) && value.truncation > 0 &&
        std::isfinite(value.maximum_weight) && value.maximum_weight > 0 &&
        std::isfinite(value.esdf_distance) && value.esdf_distance > 0;
}
struct checkpoint_header
{
    std::uint64_t magic{0x3150414d53564f43ULL};
    std::uint32_t version{1}, layout{}, esdf_schema{}, fields{}, tile_voxels{512};
    submap_record record;
    checkpoint_parameters parameters;
};
struct checkpoint_writer
{
    std::vector<std::byte> bytes;
    template<class t> void scalar(const t& value)
    {
        auto encoded = std::bit_cast<std::array<std::byte, sizeof(t)>>(value);
        bytes.insert(bytes.end(), encoded.begin(), encoded.end());
    }
};
struct checkpoint_reader
{
    std::span<const std::byte> bytes;
    std::size_t position{};
    bool good{true};
    template<class t> void scalar(t& value)
    {
        if (!good || sizeof(t) > bytes.size() - position) { good = false; return; }
        std::array<std::byte, sizeof(t)> encoded;
        std::copy_n(bytes.begin() + position, sizeof(t), encoded.begin());
        value = std::bit_cast<t>(encoded);
        position += sizeof(t);
    }
    std::size_t remaining() const { return bytes.size() - position; }
};
inline std::uint64_t checkpoint_hash(std::span<const std::byte> bytes)
{
    std::uint64_t hash{14695981039346656037ULL};

    for (auto value : bytes) { hash ^= std::to_integer<unsigned char>(value); hash *= 1099511628211ULL; }
    return hash;
}
template<class archive> void archive_header(archive& stream, checkpoint_header& header)
{
    stream.scalar(header.magic); stream.scalar(header.version); stream.scalar(header.layout);
    stream.scalar(header.esdf_schema); stream.scalar(header.fields); stream.scalar(header.tile_voxels);
    auto& record = header.record;
    stream.scalar(record.id);
    stream.scalar(record.versions.descriptor); stream.scalar(record.versions.allocation);
    stream.scalar(record.versions.tsdf); stream.scalar(record.versions.esdf);
    stream.scalar(record.versions.esdf_source); stream.scalar(record.versions.pose);
    stream.scalar(record.initial_pose.x); stream.scalar(record.initial_pose.y);
    stream.scalar(record.initial_pose.z); stream.scalar(record.initial_pose.yaw);
    stream.scalar(record.optimized_pose.x); stream.scalar(record.optimized_pose.y);
    stream.scalar(record.optimized_pose.z); stream.scalar(record.optimized_pose.yaw);
    stream.scalar(header.parameters.voxel_size); stream.scalar(header.parameters.truncation);
    stream.scalar(header.parameters.maximum_weight); stream.scalar(header.parameters.esdf_distance);
    stream.scalar(header.parameters.algorithm);
}
template<storage::schema s, storage::layout l> void encode_field(checkpoint_writer& output, const field_pool<s, l>& field)
{
    output.scalar(std::uint64_t(field.payload.size()));

    for (const auto& slot : field.index.records()) {
        output.scalar(slot.key.x); output.scalar(slot.key.y); output.scalar(slot.key.z);
        output.scalar(slot.generation); output.scalar(std::uint32_t(slot.occupied));
    }
    const auto view = field.payload.view();

    for (std::size_t tile = 0; tile < view.size(); ++tile)
        for (std::size_t voxel = 0; voxel < 512; ++voxel)
            [&]<std::size_t... f>(std::index_sequence<f...>) {
                (output.scalar(field.payload.template scalar<f>(tile, voxel)), ...);
            }(std::make_index_sequence<storage::field_count<s>>{});
}
template<storage::layout l, storage::schema es> operation_result<std::vector<std::byte>> encode_checkpoint(
    const map_snapshot<l, es>& snapshot, checkpoint_parameters parameters, std::size_t maximum_bytes)
{
    const auto& record = snapshot.record;
    const auto fields = std::uint32_t(bool(snapshot.tsdf)) | (std::uint32_t(bool(snapshot.esdf)) << 1);
    if (record.state != lifecycle::finalized || record.writer != writer_kind::none || !fields ||
        !valid_parameters(parameters) || !finite_pose(record.initial_pose) || !finite_pose(record.optimized_pose) ||
        !record.versions.pose || !record.versions.descriptor || !record.versions.allocation)
        return std::unexpected(operation_error{"checkpoint requires a finalized valid snapshot"});
    if (bool(record.versions.tsdf) != bool(snapshot.tsdf) || bool(record.versions.esdf) != bool(snapshot.esdf) ||
        (snapshot.esdf && record.versions.esdf_source != record.versions.tsdf))
        return std::unexpected(operation_error{"checkpoint snapshot is incomplete or stale"});
    checkpoint_header header;
    header.layout = std::uint32_t(l); header.esdf_schema = std::uint32_t(es);
    header.fields = fields; header.record = record; header.parameters = parameters;
    checkpoint_writer output;
    archive_header(output, header);
    std::size_t required = output.bytes.size() + 8;
    const auto count = [&]<storage::schema s>(const std::shared_ptr<const field_pool<s, l>>& field) -> operation_status {
        if (!field) return {};
        constexpr auto stride = 24 + 512 * storage::field_count<s> * 4;
        const auto epoch = s == storage::schema::tsdf ? record.versions.tsdf : record.versions.esdf;
        if (field->index.submap() != record.id || field->content_epoch != epoch ||
            field->payload.size() != field->index.slot_count() || required > maximum_bytes ||
            maximum_bytes - required < 8 || field->payload.size() > (maximum_bytes - required - 8) / stride)
            return std::unexpected(operation_error{"checkpoint field or byte budget is invalid"});
        required += 8 + field->payload.size() * stride;
        return {};
    };
    auto checked = count(snapshot.tsdf);
    if (!checked) return std::unexpected(checked.error());
    checked = count(snapshot.esdf);
    if (!checked) return std::unexpected(checked.error());
    output.bytes.reserve(required);
    if (snapshot.tsdf) encode_field(output, *snapshot.tsdf);
    if (snapshot.esdf) encode_field(output, *snapshot.esdf);
    output.scalar(checkpoint_hash(output.bytes));
    return std::move(output.bytes);
}
}
