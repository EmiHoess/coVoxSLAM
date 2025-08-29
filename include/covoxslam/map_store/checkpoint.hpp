#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/map_store/snapshot.hpp>
#include <bit>
#include <filesystem>
#include <fstream>
#include <atomic>
#include <chrono>

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
template<storage::schema s, storage::layout l> operation_result<std::shared_ptr<const field_pool<s, l>>> decode_field(
    checkpoint_reader& input, std::uint64_t id, std::uint64_t epoch, std::size_t maximum_tiles, memory_budget& budget)
{
    std::uint64_t count{};
    input.scalar(count);
    constexpr auto wire_stride = 24 + 512 * storage::field_count<s> * 4;
    constexpr auto storage_stride = storage::payload_tile_bytes<s, l>;
    if (!input.good || !epoch || count > maximum_tiles || count > input.remaining() / wire_stride ||
        count > std::numeric_limits<std::uint32_t>::max() || count > std::numeric_limits<std::size_t>::max() / storage_stride)
        return std::unexpected(operation_error{"checkpoint field length is invalid"});
    auto memory = budget.reserve({std::size_t(count) * storage_stride, 0, 0});
    if (!memory) return std::unexpected(memory.error());
    auto field = std::make_shared<field_pool<s, l>>();
    field->memory = std::move(*memory);
    std::vector<slot_record> slots(count);

    for (auto& slot : slots) {
        std::uint32_t occupied{};
        slot.key.submap = id;
        input.scalar(slot.key.x); input.scalar(slot.key.y); input.scalar(slot.key.z);
        input.scalar(slot.generation); input.scalar(occupied);
        if (occupied > 1) return std::unexpected(operation_error{"checkpoint slot state is invalid"});
        slot.occupied = bool(occupied);
    }
    auto checked = field->index.restore(id, s == storage::schema::tsdf ? field_kind::tsdf : field_kind::esdf, slots);
    if (!checked) return std::unexpected(checked.error());
    checked = field->payload.initialize(count);
    if (!checked) return std::unexpected(checked.error());

    for (std::size_t tile = 0; tile < count; ++tile)
        for (std::size_t voxel = 0; voxel < 512; ++voxel)
            [&]<std::size_t... f>(std::index_sequence<f...>) {
                (input.scalar(field->payload.template scalar<f>(tile, voxel)), ...);
            }(std::make_index_sequence<storage::field_count<s>>{});
    if (!input.good) return std::unexpected(operation_error{"checkpoint field is truncated"});
    field->content_epoch = epoch;
    return field;
}
template<storage::layout l, storage::schema es> operation_result<map_snapshot<l, es>> decode_checkpoint(
    std::span<const std::byte> bytes, checkpoint_parameters expected, std::size_t maximum_tiles, memory_budget& budget)
{
    if (bytes.size() < 8) return std::unexpected(operation_error{"checkpoint is truncated"});
    checkpoint_reader trailer{bytes.last(8)};
    std::uint64_t hash{};
    trailer.scalar(hash);
    if (hash != checkpoint_hash(bytes.first(bytes.size() - 8)))
        return std::unexpected(operation_error{"checkpoint checksum differs"});
    checkpoint_reader input{bytes.first(bytes.size() - 8)};
    checkpoint_header header;
    archive_header(input, header);
    if (!input.good || header.magic != 0x3150414d53564f43ULL || header.version != 1 || header.tile_voxels != 512 ||
        header.layout > std::uint32_t(storage::layout::pool_soa) || header.esdf_schema != std::uint32_t(es) ||
        !header.fields || (header.fields & ~3U) || !valid_parameters(header.parameters) || header.parameters != expected)
        return std::unexpected(operation_error{"checkpoint format or configuration differs"});
    const auto& record = header.record;
    if (!record.versions.pose || !record.versions.allocation || !record.versions.descriptor ||
        !finite_pose(record.initial_pose) || !finite_pose(record.optimized_pose) ||
        bool(header.fields & 1U) != bool(record.versions.tsdf) ||
        bool(header.fields & 2U) != bool(record.versions.esdf) ||
        ((header.fields & 2U) && (!record.versions.tsdf || record.versions.esdf_source != record.versions.tsdf)) ||
        (!(header.fields & 2U) && record.versions.esdf_source))
        return std::unexpected(operation_error{"checkpoint epochs or poses are inconsistent"});
    map_snapshot<l, es> result;
    result.record = header.record;
    result.record.state = lifecycle::finalized;
    result.record.cpu = residency::resident;
    if (header.fields & 1U) {
        auto field = decode_field<storage::schema::tsdf, l>(input, header.record.id, header.record.versions.tsdf, maximum_tiles, budget);
        if (!field) return std::unexpected(field.error());
        result.tsdf = std::move(*field);
    }
    if (header.fields & 2U) {
        auto field = decode_field<es, l>(input, header.record.id, header.record.versions.esdf, maximum_tiles, budget);
        if (!field) return std::unexpected(field.error());
        result.esdf = std::move(*field);
    }
    if (!input.good || input.remaining()) return std::unexpected(operation_error{"checkpoint has trailing or missing data"});
    return result;
}
inline operation_status save_checkpoint_bytes(const std::filesystem::path& destination, std::span<const std::byte> bytes)
{
    if (destination.filename().empty() || bytes.empty() || bytes.size() > std::size_t(std::numeric_limits<std::streamsize>::max()))
        return std::unexpected(operation_error{"invalid checkpoint destination or size"});
    std::error_code error;
    if (std::filesystem::exists(destination, error) || error)
        return std::unexpected(operation_error{"checkpoint destination already exists or cannot be inspected"});
    static std::atomic<std::uint64_t> sequence{};
    std::filesystem::path temporary;
    bool created{};

    for (unsigned attempt = 0; attempt < 64 && !created; ++attempt) {
        temporary = destination;
        temporary += ".pending." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
            "." + std::to_string(sequence.fetch_add(1));
        created = std::filesystem::create_directory(temporary, error);
        if (error) return std::unexpected(operation_error{"cannot create checkpoint staging directory"});
    }
    if (!created) return std::unexpected(operation_error{"checkpoint staging names exhausted"});
    struct cleanup
    {
        std::filesystem::path directory;
        bool armed{true};
        ~cleanup()
        {
            if (!armed) return;
            std::error_code ignored;
            std::filesystem::remove(directory / "map.bin", ignored);
            std::filesystem::remove(directory, ignored);
        }
    } pending{temporary};
    std::ofstream file(temporary / "map.bin", std::ios::binary | std::ios::trunc);
    if (!file) return std::unexpected(operation_error{"cannot open checkpoint output"});
    file.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    file.flush();
    if (!file) return std::unexpected(operation_error{"cannot finish checkpoint output"});
    file.close();
    if (!file) return std::unexpected(operation_error{"cannot close checkpoint output"});
    std::filesystem::rename(temporary, destination, error);
    if (error) return std::unexpected(operation_error{"cannot publish checkpoint generation"});
    pending.armed = false;
    return {};
}
inline operation_result<std::vector<std::byte>> read_checkpoint_bytes(const std::filesystem::path& directory, std::size_t maximum_bytes)
{
    std::error_code error;
    const auto path = directory / "map.bin";
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > maximum_bytes || size > std::uintmax_t(std::numeric_limits<std::streamsize>::max()))
        return std::unexpected(operation_error{"checkpoint size is unavailable or exceeds its budget"});
    std::vector<std::byte> bytes(std::size_t(size), std::byte{});
    std::ifstream file(path, std::ios::binary);
    if (!file || !file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(size)) || file.peek() != std::char_traits<char>::eof())
        return std::unexpected(operation_error{"checkpoint read is incomplete or changed"});
    return bytes;
}
}
