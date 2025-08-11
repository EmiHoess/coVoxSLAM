#include <covoxslam/dataset.hpp>
#include "json_reader.hpp"
#include "dataset_metadata.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace covoxslam
{
namespace detail
{
operation_result<pose> read_pose(const json_value& value)
{
    const auto translation_member = value.member("translation_m");
    if (!translation_member) return std::unexpected(translation_member.error());
    const auto rotation_member = value.member("quaternion_xyzw");
    if (!rotation_member) return std::unexpected(rotation_member.error());
    const auto& translation = translation_member->get();
    const auto& rotation = rotation_member->get();
    if (translation.kind != json_kind::array || translation.items.size() != 3 ||
        rotation.kind != json_kind::array || rotation.items.size() != 4)
        return std::unexpected(operation_error{"invalid pose dimensions"});
     auto scalar = [](const json_value& item) -> operation_result<float>
    {
        const auto number = item.number();
        if (!number) return std::unexpected(number.error());
        if (std::abs(*number) > std::numeric_limits<float>::max())
            return std::unexpected(operation_error{"pose component exceeds float range"});
        return static_cast<float>(*number);
    };
    std::array<float, 7> components;
    for (std::size_t index = 0; index < translation.items.size(); ++index) {
        const auto value = scalar(translation.items[index]);
        if (!value) return std::unexpected(value.error());
        components[index] = *value;
    }
    for (std::size_t index = 0; index < rotation.items.size(); ++index) {
        const auto value = scalar(rotation.items[index]);
        if (!value) return std::unexpected(value.error());
        components[index + 3] = *value;
    }
    pose result;
    result.translation = {components[0], components[1], components[2]};
    result.rotation = {components[3], components[4], components[5], components[6]};
    const auto& q = result.rotation;
    const double norm = std::hypot(std::hypot(double(q.x), double(q.y)), std::hypot(double(q.z), double(q.w)));
    if (std::abs(norm - 1) > 0.001)
        return std::unexpected(operation_error{"pose quaternion is not normalized"});
    result.rotation = normalized(result.rotation);
     return result;
}

pose legacy_read_pose(const json_value& value)
{
    const auto& translation = value.legacy_at("translation_m");
    const auto& rotation = value.legacy_at("quaternion_xyzw");
    if (translation.kind != json_kind::array || translation.items.size() != 3 ||
        rotation.kind != json_kind::array || rotation.items.size() != 4)
        throw std::runtime_error("invalid pose dimensions");
     auto scalar = [](const json_value& item)
    {
        const double number = item.legacy_number();
        if (std::abs(number) > std::numeric_limits<float>::max())
            throw std::runtime_error("pose component exceeds float range");
        return static_cast<float>(number);
    };
    pose result;
    result.translation = {scalar(translation.items[0]), scalar(translation.items[1]), scalar(translation.items[2])};
    result.rotation = {scalar(rotation.items[0]), scalar(rotation.items[1]), scalar(rotation.items[2]), scalar(rotation.items[3])};
    const auto& q = result.rotation;
    const double norm = std::hypot(std::hypot(double(q.x), double(q.y)), std::hypot(double(q.z), double(q.w)));
    if (std::abs(norm - 1) > 0.001) throw std::runtime_error("pose quaternion is not normalized");
    result.rotation = normalized(result.rotation);
     return result;
}
}

namespace
{
payload_cache_descriptor hash_file(const std::filesystem::path& path);
std::filesystem::path frame_buffer_path(
    const detail::json_value& buffers, std::string_view name,
    std::string_view dtype, std::string_view layout,
    std::initializer_list<std::size_t> shape,
    const std::filesystem::path& root,
    std::string_view unit = {}, std::string_view encoding = {})
{
    const auto& buffer = buffers.legacy_at(name);
    const auto& shape_value = buffer.legacy_at("shape");
    const auto& dimensions = shape_value.items;
    if (shape_value.kind != detail::json_kind::array || buffer.legacy_at("dtype").legacy_string() != dtype ||
        buffer.legacy_at("layout").legacy_string() != layout ||
         buffer.legacy_at("byte_order").legacy_string() != "little_endian" ||
        dimensions.size() != shape.size())
        throw std::runtime_error("unsupported organized buffer format");
    if (!unit.empty() && buffer.legacy_at("unit").legacy_string() != unit)
        throw std::runtime_error("unsupported frame buffer unit");
    if (!encoding.empty() && buffer.members.contains("invalid_encoding") &&
        buffer.legacy_at("invalid_encoding").legacy_string() != encoding)
        throw std::runtime_error("unsupported frame zero encoding");
    std::size_t index{};
    for (const auto extent : shape) {
        if (dimensions[index++].legacy_unsigned_integer() != extent)
            throw std::runtime_error("buffer shape does not match the frame");
    }
    return detail::legacy_dataset_path(root, buffer.legacy_at("file").legacy_string());
}

std::size_t frame_beam_count(const frame_descriptor& frame)
{
    constexpr std::size_t maximum = 16U * 1024U * 1024U;
    if (frame.rows == 0 || frame.columns == 0 ||
        frame.rows > maximum / frame.columns)
        throw std::runtime_error("frame dimensions exceed the supported bounds");
    const auto count = frame.rows * frame.columns;
    if (frame.beam_count != count || frame.end_stamp_ns <= frame.start_stamp_ns)
        throw std::runtime_error("invalid frame beam count or scan interval");
     return count;
}
}

operation_result<frame_descriptor> read_frame_descriptor(
    const std::filesystem::path& root, const std::filesystem::path& metadata_path)
{
    frame_descriptor frame;
    frame.metadata_path = detail::legacy_dataset_path(root, metadata_path.lexically_relative(root));
    frame.metadata_content_hash = hash_file(frame.metadata_path).content_hash;
    const auto parsed = detail::read_metadata(frame.metadata_path);
    if (!parsed) return std::unexpected(parsed.error());
    const auto& value = *parsed;
    if (value.legacy_at("schema_version").legacy_string() != "covox.spinning_lidar.v2")
        return std::unexpected(operation_error{"unsupported frame schema"});
    if (value.legacy_at("document_type").legacy_string() != "spinning_lidar_frame")
        return std::unexpected(operation_error{"unexpected frame document type"});
    const auto id = value.legacy_at("frame_id").legacy_unsigned_integer();
    if (id > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        return std::unexpected(operation_error{"frame identifier exceeds the integer range"});
     frame.frame_id = static_cast<int>(id);
    if (value.members.contains("submap_id")) {
        const double id_value = value.legacy_at("submap_id").legacy_number();
        if (id_value < -1 || id_value > std::numeric_limits<int>::max() ||
            std::trunc(id_value) != id_value)
            return std::unexpected(operation_error{"invalid submap identifier"});
        frame.submap_id = static_cast<int>(id_value);
    }
    frame.start_stamp_ns = value.legacy_at("header_stamp_ns").legacy_unsigned_integer();
    frame.end_stamp_ns = value.legacy_at("scan_end_stamp_ns").legacy_unsigned_integer();
    frame.rows = value.legacy_at("organized_rows").legacy_unsigned_integer();
    frame.columns = value.legacy_at("organized_cols").legacy_unsigned_integer();
     frame.beam_count = value.legacy_at("beam_count").legacy_unsigned_integer();
    frame.hit_count = value.legacy_at("hit_count").legacy_unsigned_integer();
    frame.dropout_count = value.legacy_at("dropout_count").legacy_unsigned_integer();
    frame.invalid_count = value.legacy_at("invalid_count").legacy_unsigned_integer();
    frame.outlier_count = value.legacy_at("outlier_count").legacy_unsigned_integer();
    const auto count = frame_beam_count(frame);
    if (frame.hit_count > count || frame.dropout_count > count - frame.hit_count ||
        frame.invalid_count != count - frame.hit_count - frame.dropout_count ||
        frame.outlier_count > frame.hit_count)
        return std::unexpected(operation_error{"inconsistent frame counters"});
    frame.ground_truth_start = detail::legacy_read_pose(value.legacy_at("pose_world_lidar_start"));
    frame.ground_truth_end = detail::legacy_read_pose(value.legacy_at("pose_world_lidar_end"));
    frame.odometry_start = value.members.contains("odometry_pose_world_lidar_start")
        ? detail::legacy_read_pose(value.legacy_at("odometry_pose_world_lidar_start"))
        : frame.ground_truth_start;
     frame.odometry_end = value.members.contains("odometry_pose_world_lidar_end")
        ? detail::legacy_read_pose(value.legacy_at("odometry_pose_world_lidar_end"))
        : frame.ground_truth_end;
    const auto& buffers = value.legacy_at("buffers");
    const auto rows = frame.rows;
    const auto columns = frame.columns;
    frame.raw_xyz_path = frame_buffer_path(
        buffers, "raw_xyz", "float32", "ring_major", {rows, columns, 3}, root, {}, "zero_xyz");
    if (buffers.members.contains("deskewed_xyz"))
        frame.deskewed_xyz_path = frame_buffer_path(
            buffers, "deskewed_xyz", "float32", "ring_major", {rows, columns, 3}, root, {}, "zero_xyz");
    frame.range_path = frame_buffer_path(
        buffers, "range", "uint32", "ring_major", {rows, columns}, root, "millimeters", "zero");
    frame.status_path = frame_buffer_path(
        buffers, "status", "uint8", "ring_major", {rows, columns}, root);
    frame.time_offset_ns_path = frame_buffer_path(
        buffers, "time_offset_ns", "uint32", "one_per_column", {columns}, root, "nanoseconds");
    return frame;
}

namespace
{
operation_status load_json_trajectories(dataset& input, const detail::json_value& metadata);
void validate_trajectory_coverage(const dataset& input);
operation_status load_dataset_inputs(dataset& input, const detail::json_value& metadata,
                                     dataset_load_options options);
std::uint64_t input_structure_hash(const dataset& input, const detail::json_value& metadata);
void index_dataset_payloads(dataset& input, std::uint64_t structure_hash);
void write_input_cache(const dataset& input, std::uint64_t structure_hash);
void verify_frame_bytes(const dataset& input, const std::filesystem::path& path,
                        std::span<const std::uint8_t> bytes);
}

operation_result<dataset> load_dataset(const std::filesystem::path& root, dataset_load_options options)
{
    using namespace detail;
    dataset result;
    std::error_code path_error;
    auto selected = std::filesystem::canonical(root, path_error);
    if (path_error) return std::unexpected(operation_error{"cannot resolve dataset path"});
    const auto directory = std::filesystem::is_directory(selected, path_error);
    if (path_error) return std::unexpected(operation_error{"cannot inspect dataset path"});
    result.root = directory ? selected : selected.parent_path();
    if (directory) {
        const bool json_exists = std::filesystem::exists(selected / "dataset.json", path_error);
        if (path_error) return std::unexpected(operation_error{"cannot select dataset manifest"});
        selected /= json_exists ? "dataset.json" : "dataset.toml";
    }
    result.manifest_path = selected;
    auto parsed = read_metadata(selected);
    if (!parsed) return std::unexpected(parsed.error());
    const auto& metadata = *parsed;
    const auto manifest_hash = hash_file(selected).content_hash;
    const auto encoding = selected.extension() == ".toml" ? "toml" : "json";
    if (metadata.members.contains("metadata_format") && metadata.legacy_at("metadata_format").legacy_string() != encoding)
        return std::unexpected(operation_error{"metadata format disagrees with extension"});
    result.schema_version = metadata.legacy_at("schema_version").legacy_string();
    if (result.schema_version != "covox.spinning_lidar.v2" ||
        metadata.legacy_at("document_type").legacy_string() != "spinning_lidar_dataset")
        return std::unexpected(operation_error{"unsupported dataset schema"});
    result.name = metadata.legacy_at("name").legacy_string();
     result.type = metadata.legacy_at("type").legacy_string();
    const auto& units = metadata.legacy_at("units");
    if (units.legacy_at("distance").legacy_string() != "meters" || units.legacy_at("angle").legacy_string() != "radians" ||
        units.legacy_at("timestamp").legacy_string() != "nanoseconds" || units.legacy_at("range").legacy_string() != "millimeters")
        return std::unexpected(operation_error{"unsupported dataset units"});
    const auto sensor_path = legacy_dataset_path(result.root, metadata.legacy_at("sensor_file").legacy_string());
    const auto sensor_hash = hash_file(sensor_path).content_hash;
    const auto parsed_calibration = read_metadata(sensor_path);
    if (!parsed_calibration) return std::unexpected(parsed_calibration.error());
    const auto& calibration = *parsed_calibration;
    if (calibration.legacy_at("schema_version").legacy_string() != result.schema_version ||
        calibration.legacy_at("document_type").legacy_string() != "spinning_lidar_sensor")
        return std::unexpected(operation_error{"unsupported sensor schema"});
    const auto& convention = metadata.legacy_at("frame_convention");
    if (convention.legacy_at("handedness").legacy_string() != "right_handed" ||
        convention.legacy_at("quaternion_order").legacy_string() != "xyzw" ||
        convention.legacy_at("transform_notation").legacy_string() != "T_parent_child_maps_child_to_parent")
        return std::unexpected(operation_error{"unsupported coordinate convention"});
    const auto& timing = metadata.legacy_at("timestamp_convention");
    if (timing.legacy_at("origin").legacy_string() != "dataset_relative" ||
        timing.legacy_at("scan_header").legacy_string() != "first_column" ||
        timing.legacy_at("beam_offsets").legacy_string() != "relative_to_scan_header")
        return std::unexpected(operation_error{"unsupported timestamp convention"});
    auto& sensor = result.sensor;
    sensor.model = calibration.legacy_at("sensor_model").legacy_string();
    sensor.acquisition_model = calibration.legacy_at("acquisition_model").legacy_string();
    sensor.rings = calibration.legacy_at("rings").legacy_unsigned_integer();
    sensor.columns = calibration.legacy_at("columns").legacy_unsigned_integer();
    if (!sensor.rings || !sensor.columns || sensor.rings > 16777216 / sensor.columns)
        return std::unexpected(operation_error{"invalid sensor dimensions"});
    if (calibration.legacy_at("beam_count").legacy_unsigned_integer() != sensor.rings * sensor.columns)
        return std::unexpected(operation_error{"sensor beam count mismatch"});
     sensor.scan_rate_hz = calibration.legacy_at("scan_rate_hz").legacy_number();
    sensor.scan_period_ns = calibration.legacy_at("scan_period_ns").legacy_unsigned_integer();
    sensor.azimuth_start_rad = calibration.legacy_at("azimuth_start_rad").legacy_number();
    sensor.azimuth_coverage_rad = calibration.legacy_at("azimuth_coverage_rad").legacy_number();
    const auto direction = calibration.legacy_at("rotation_direction").legacy_string();
    if (direction != "clockwise" && direction != "counter_clockwise")
        return std::unexpected(operation_error{"unsupported sensor rotation direction"});
    sensor.counter_clockwise = direction == "counter_clockwise";
    const auto& angles = calibration.legacy_at("vertical_angles_rad");
    if (angles.kind != json_kind::array || angles.items.size() != sensor.rings)
        return std::unexpected(operation_error{"invalid vertical angle table"});
    for (const auto& angle : angles.items)
    {
        const auto radians = angle.legacy_number();
         if (std::abs(radians) > 1.5707963267948966) return std::unexpected(operation_error{"invalid beam elevation"});
        sensor.vertical_angles_rad.push_back(static_cast<float>(radians));
    }
    const auto minimum = calibration.legacy_at("min_range_m").legacy_number();
    const auto maximum = calibration.legacy_at("max_range_m").legacy_number();
    if (minimum < 0 || maximum <= minimum || maximum > std::numeric_limits<float>::max() ||
        sensor.scan_rate_hz <= 0 || !sensor.scan_period_ns || sensor.azimuth_coverage_rad <= 0 ||
        sensor.azimuth_coverage_rad > 6.283185307179587)
        return std::unexpected(operation_error{"invalid sensor ranges or timing"});
    sensor.min_range_m = static_cast<float>(minimum);
    sensor.max_range_m = static_cast<float>(maximum);
    sensor.body_from_lidar = legacy_read_pose(calibration.legacy_at("T_parent_child"));
    const std::filesystem::path pattern(metadata.legacy_at("frame_metadata_pattern").legacy_string());
     const auto frame_directory = legacy_dataset_path(result.root, pattern.parent_path());
    const auto filename = pattern.filename().string();
    const auto wildcard = filename.find('*');
    if (wildcard == std::string::npos || filename.find('*', wildcard + 1) != std::string::npos)
        return std::unexpected(operation_error{"expected one frame metadata wildcard"});
    for (const auto& entry : std::filesystem::directory_iterator(frame_directory))
    {
        const auto name = entry.path().filename().string();
        if (entry.is_regular_file() && name.starts_with(filename.substr(0, wildcard)) &&
            name.ends_with(filename.substr(wildcard + 1)))
        {
            auto frame = read_frame_descriptor(result.root,
                legacy_dataset_path(result.root, pattern.parent_path() / entry.path().filename()));
            if (!frame) return std::unexpected(frame.error());
            if (frame->rows != sensor.rings || frame->columns != sensor.columns)
                return std::unexpected(operation_error{"frame dimensions differ from the sensor calibration"});
            result.frames.push_back(std::move(*frame));
        }
    }
     if (result.frames.size() != metadata.legacy_at("num_frames").legacy_unsigned_integer())
        return std::unexpected(operation_error{"frame metadata count mismatch"});
    std::sort(result.frames.begin(), result.frames.end(), [](const auto& left, const auto& right) {
        return left.frame_id < right.frame_id;
    });
    for (std::size_t index = 1; index < result.frames.size(); ++index)
        if (result.frames[index - 1].frame_id == result.frames[index].frame_id)
            return std::unexpected(operation_error{"duplicate frame identifier"});
    auto inputs = load_dataset_inputs(result, metadata, options);
    if (!inputs) return std::unexpected(inputs.error());
    if (manifest_hash != hash_file(selected).content_hash ||
        sensor_hash != hash_file(sensor_path).content_hash)
        return std::unexpected(operation_error{"recording metadata changed while inputs were loaded"});
    if (options.update_cache && result.input_source == dataset_input_source::full_scan)
        write_input_cache(result, input_structure_hash(result, metadata));
    return result;
}
}

#include <bit>
#include <fstream>

namespace covoxslam
{
namespace
{
std::vector<std::uint8_t> read_frame_bytes(
    const dataset& input, const std::filesystem::path& path,
    std::size_t count, std::size_t width, input_timing& timing)
{
    const auto started = std::chrono::steady_clock::now();
    if (path.empty() || width == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / width)
        throw std::runtime_error("missing frame buffer or invalid byte count");
    const auto resolved = detail::legacy_dataset_path(input.root, path.lexically_relative(input.root));
    const auto bytes = count * width;
     std::ifstream file(resolved, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(bytes))
        throw std::runtime_error("frame buffer byte count does not match its shape");
    file.seekg(0);
    std::vector<std::uint8_t> result(bytes);
    file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
    if (!file || file.gcount() != static_cast<std::streamsize>(bytes) ||
        file.peek() != std::char_traits<char>::eof() || file.bad())
        throw std::runtime_error("frame buffer could not be read completely");
    const auto loaded = std::chrono::steady_clock::now();
    timing.read_ms += std::chrono::duration<double, std::milli>(loaded - started).count();
    timing.payload_bytes += bytes;
    verify_frame_bytes(input, resolved, result);
    timing.validation_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loaded).count();
    return result;
}

std::uint32_t decode_frame_u32(const std::uint8_t* bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8U) |
         (static_cast<std::uint32_t>(bytes[2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

std::vector<std::uint32_t> read_frame_u32(
    const dataset& input, const std::filesystem::path& path, std::size_t count, input_timing& timing)
{
    const auto bytes = read_frame_bytes(input, path, count, 4, timing);
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> result(count);
        if (!bytes.empty()) std::memcpy(result.data(), bytes.data(), bytes.size());
    timing.decode_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}

std::vector<vec3f> read_frame_xyz(
    const dataset& input, const std::filesystem::path& path, std::size_t count, input_timing& timing)
{
    const auto bytes = read_frame_bytes(input, path, count, 12, timing);
    const auto started = std::chrono::steady_clock::now();
     std::vector<vec3f> result(count);
    constexpr bool native = sizeof(vec3f) == 12 &&
        std::is_trivially_copyable_v<vec3f> && offsetof(vec3f, y) == 4 && offsetof(vec3f, z) == 8;
    if constexpr (native)
        if (!bytes.empty()) std::memcpy(result.data(), bytes.data(), bytes.size());

    for (std::size_t index = 0; index < count; ++index) {
        const auto* data = bytes.data() + index * 12;
        const vec3f point = native ? result[index] : vec3f{
            std::bit_cast<float>(decode_frame_u32(data)),
            std::bit_cast<float>(decode_frame_u32(data + 4)),
            std::bit_cast<float>(decode_frame_u32(data + 8))};
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            throw std::runtime_error("frame buffer contains a non-finite coordinate");
        result[index] = point;
    }
    timing.decode_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}
}

std::size_t organized_frame::flat_index(std::size_t ring, std::size_t column) const
{
    frame_beam_count(descriptor);
    if (ring >= descriptor.rows || column >= descriptor.columns)
        throw std::out_of_range("organized frame index is out of range");
     return ring * descriptor.columns + column;
}

const vec3f& organized_frame::point(
    point_source source, std::size_t ring, std::size_t column) const
{
    const auto index = flat_index(ring, column);
    if (source == point_source::raw)
        return raw_xyz.at(index);
    if (source == point_source::deskewed)
        return deskewed_xyz.at(index);
    throw std::invalid_argument("unsupported point source");
}

std::uint64_t organized_frame::beam_stamp_ns(std::size_t column) const
{
    if (column >= descriptor.columns)
        throw std::out_of_range("organized frame column is out of range");
    const auto offset = time_offset_ns.at(column);
     if (descriptor.end_stamp_ns <= descriptor.start_stamp_ns ||
        offset >= descriptor.end_stamp_ns - descriptor.start_stamp_ns)
        throw std::runtime_error("column time offset is outside the scan interval");
    return descriptor.start_stamp_ns + offset;
}

bool organized_frame::valid_return(std::size_t ring, std::size_t column) const {
    const auto value = status.at(flat_index(ring, column));
    return value == static_cast<std::uint8_t>(beam_status::valid_hit) ||
        value == static_cast<std::uint8_t>(beam_status::valid_outlier);
}

namespace
{
struct frame_buffer_request {
    const std::filesystem::path* path;
    std::size_t count;
    std::size_t width;
     bool requested;
}
;

void validate_frame_payload_bounds(
    const dataset& input, const frame_descriptor& frame, const frame_load_options& options)
{
    if (options.max_payload_bytes == 0)
        throw std::invalid_argument("frame payload byte limit must be positive");
    const auto count = frame_beam_count(frame);
    const std::array<frame_buffer_request, 5> requests{{
        {&frame.raw_xyz_path, count, 12, options.raw_xyz},
        {&frame.deskewed_xyz_path, count, 12, options.deskewed_xyz},
        {&frame.range_path, count, 4, options.range},
         {&frame.status_path, count, 1, options.status},
        {&frame.time_offset_ns_path, frame.columns, 4, options.time_offset_ns}}};
    std::size_t total{};
    for (const auto& request : requests)
    {
        if (!request.requested)
            continue;
        if (request.count > (options.max_payload_bytes - total) / request.width)
            throw std::runtime_error("requested frame buffers exceed the payload byte limit");
        const auto bytes = request.count * request.width;
        total += bytes;
        if (request.path->empty())
            throw std::runtime_error("a requested frame buffer is absent");
        const auto path = detail::legacy_dataset_path(
            input.root, request.path->lexically_relative(input.root));
        if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != bytes)
            throw std::runtime_error("requested frame buffer has an invalid path or size");
    }
}

void validate_frame_contents(const organized_frame& frame)
{
    if (frame.status.empty())
        return;
    std::array<std::size_t, 4> counts{};
    const auto valid_point = [](vec3f point) {
        return point.x != 0 || point.y != 0 || point.z != 0;
    };
    for (std::size_t index = 0; index < frame.status.size(); ++index)
    {
        const auto status = frame.status[index];
        if (status >= counts.size())
            throw std::runtime_error("frame buffer contains an unknown status code");
        ++counts[status];
        const bool valid = status == static_cast<std::uint8_t>(beam_status::valid_hit) ||
             status == static_cast<std::uint8_t>(beam_status::valid_outlier);
        if ((!frame.range_mm.empty() && (frame.range_mm[index] != 0) != valid) ||
            (!frame.raw_xyz.empty() && valid_point(frame.raw_xyz[index]) != valid) ||
            (!frame.deskewed_xyz.empty() && valid_point(frame.deskewed_xyz[index]) != valid))
            throw std::runtime_error("frame status disagrees with zero return encoding");
    }
    const auto& descriptor = frame.descriptor;
    if (counts[0] != descriptor.invalid_count || counts[2] != descriptor.dropout_count ||
        counts[1] + counts[3] != descriptor.hit_count || counts[3] != descriptor.outlier_count)
        throw std::runtime_error("binary status counts disagree with the frame metadata");
}
}

organized_frame load_frame(
    const dataset& input, std::size_t frame_index, frame_load_options options)
{
    organized_frame frame;
    frame.descriptor = input.frames.at(frame_index);
    const auto count = frame_beam_count(frame.descriptor);
    validate_frame_payload_bounds(input, frame.descriptor, options);
    if (options.deskewed_xyz && frame.descriptor.deskewed_xyz_path.empty())
        throw std::runtime_error("the frame has no deskewed XYZ buffer");
     if (options.raw_xyz)
        frame.raw_xyz = read_frame_xyz(input, frame.descriptor.raw_xyz_path, count, frame.timing);
    if (options.deskewed_xyz)
        frame.deskewed_xyz = read_frame_xyz(input, frame.descriptor.deskewed_xyz_path, count, frame.timing);
    if (options.range)
        frame.range_mm = read_frame_u32(input, frame.descriptor.range_path, count, frame.timing);
    if (options.status)
        frame.status = read_frame_bytes(input, frame.descriptor.status_path, count, 1, frame.timing);
    if (options.time_offset_ns)
        frame.time_offset_ns = read_frame_u32(
            input, frame.descriptor.time_offset_ns_path, frame.descriptor.columns, frame.timing);
    const auto validation_started = std::chrono::steady_clock::now();
    for (std::size_t column = 0; column < frame.time_offset_ns.size(); ++column)
    {
        frame.beam_stamp_ns(column);
        if (column != 0 && frame.time_offset_ns[column] <= frame.time_offset_ns[column - 1])
            throw std::runtime_error("column time offsets are not strictly increasing");
    }
    for (const auto value : frame.status) {
        if (value > static_cast<std::uint8_t>(beam_status::valid_outlier))
            throw std::runtime_error("frame buffer contains an unknown status code");
    }
    validate_frame_contents(frame);
    frame.timing.validation_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - validation_started).count();
    return frame;
}
}

#include <array>
#include <cstring>
#include <fstream>
#include <span>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace covoxslam
{
namespace
{
constexpr std::uint64_t hash_basis = 14695981039346656037ULL;
constexpr std::uint64_t cache_magic = 0x31584449434f5658ULL;
constexpr std::size_t maximum_cache_bytes = 1024U * 1024U * 1024U;
 constexpr std::size_t maximum_trajectory_samples = 16U * 1024U * 1024U;

std::uint64_t hash_bytes(std::span<const std::uint8_t> bytes, std::uint64_t hash = hash_basis)
{
    for (const auto byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

cache_file_identity file_identity(const std::filesystem::path& path)
{
    if (!std::filesystem::is_regular_file(path))
        throw std::runtime_error("recording input is not a regular file");
    const auto modified = std::filesystem::last_write_time(path).time_since_epoch().count();
    return {std::filesystem::file_size(path), static_cast<std::int64_t>(modified)};
}

payload_cache_descriptor hash_file(const std::filesystem::path& path)
{
    const auto before = file_identity(path);
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open recording input");
    std::array<std::uint8_t, 65536> bytes;
    std::uint64_t hash = hash_basis;
     std::uint64_t count{};
    while (file)
    {
        file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        const auto size = static_cast<std::size_t>(file.gcount());
        hash = hash_bytes(std::span(bytes).first(size), hash);
        count += size;
    }
    if (!file.eof() || file.bad() || count != before.size || file_identity(path) != before)
        throw std::runtime_error("recording input changed or could not be read");
    return {before, hash};
}

class cache_buffer
{
    std::size_t position{};
public:
    std::vector<std::uint8_t> bytes;
    template<class value_type>
    void append(value_type value)
    {
        if (bytes.size() > maximum_cache_bytes - sizeof(value_type))
            throw std::length_error("input cache exceeds byte limit");
        const auto* first = reinterpret_cast<const std::uint8_t*>(&value);
         bytes.insert(bytes.end(), first, first + sizeof(value_type));
    }
    template<class value_type>
    value_type take() {
        if (sizeof(value_type) > bytes.size() - position)
            throw std::runtime_error("truncated input cache");
        value_type value;
        std::memcpy(&value, bytes.data() + position, sizeof(value_type));
        position += sizeof(value_type);
        return value;
    }
    std::size_t remaining() const
    {
        return bytes.size() - position;
    }
}
;

cache_buffer read_cache_bytes(const std::filesystem::path& path)
{
    const auto size = std::filesystem::file_size(path);
    if (size < 16 || size > maximum_cache_bytes)
        throw std::runtime_error("invalid input cache length");
    cache_buffer result;
     result.bytes.resize(static_cast<std::size_t>(size));
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(result.bytes.data()), static_cast<std::streamsize>(size)) ||
        file.peek() != std::char_traits<char>::eof() || file.bad())
        throw std::runtime_error("cannot read complete input cache");
    std::uint64_t checksum;
    std::memcpy(&checksum, result.bytes.data() + size - 8, 8);
    result.bytes.resize(static_cast<std::size_t>(size) - 8);
    if (hash_bytes(result.bytes) != checksum || result.take<std::uint64_t>() != cache_magic)
        throw std::runtime_error("input cache checksum or native format mismatch");
    return result;
}

void write_cache_bytes(const std::filesystem::path& root, cache_buffer& buffer)
{
    const auto directory = root / ".covox_input_cache_write";
    std::error_code error;
    if (!std::filesystem::create_directory(directory, error)) return;
    const auto temporary = directory / "input.bin";
    const auto destination = root / ".covox_input_cache_v1.bin";
    try {
        buffer.append(hash_bytes(buffer.bytes));
        std::ofstream file(temporary, std::ios::binary);
        file.write(reinterpret_cast<const char*>(buffer.bytes.data()),
                    static_cast<std::streamsize>(buffer.bytes.size()));
        file.close();
        if (file)
        {
#ifdef _WIN32
            MoveFileExW(temporary.c_str(), destination.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH);
#else
             std::filesystem::rename(temporary, destination, error);
#endif
        }
    }
    catch (const std::exception&)
    {}
    std::filesystem::remove(temporary, error);
    std::filesystem::remove(directory, error);
}
}
}

namespace covoxslam
{
namespace
{
void validate_pose_sample(const pose_sample& sample)
{
    const auto& position = sample.pose.translation;
    const auto& rotation = sample.pose.rotation;
    const double norm = std::hypot(std::hypot(double(rotation.x), double(rotation.y)),
                                  std::hypot(double(rotation.z), double(rotation.w)));
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z) ||
         !std::isfinite(norm) || std::abs(norm - 1) > 0.001)
        throw std::runtime_error("invalid trajectory pose");
}

operation_result<std::vector<pose_sample>> read_trajectory(
    const std::filesystem::path& path, std::string_view kind, const detail::json_value& convention)
{
    const auto before = file_identity(path);
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open trajectory");
    std::vector<pose_sample> result;
    std::vector<char> line(1024U * 1024U + 2);
    std::size_t line_number{};
    while (file.getline(line.data(), static_cast<std::streamsize>(line.size())))
    {
        ++line_number;
        const auto count = static_cast<std::size_t>(file.gcount()) - (file.eof() ? 0 : 1);
        if (count == 0 || (count == 1 && line[0] == '\r')) continue;
        if (count > 1024U * 1024U || result.size() >= maximum_trajectory_samples)
            throw std::length_error("trajectory exceeds its line or sample limit");
        auto parsed = detail::parse_json(std::string_view(line.data(), count));
        if (!parsed) {
            parsed.error().message = path.string() + ": line " + std::to_string(line_number) + ": " +
                                     parsed.error().message;
            return std::unexpected(parsed.error());
        }
        const auto& value = *parsed;
        if (value.legacy_at("schema_version").legacy_string() != "covox.spinning_lidar.v2" ||
             value.legacy_at("document_type").legacy_string() != "spinning_lidar_trajectory_record")
            throw std::runtime_error("unsupported trajectory record schema");
        if (value.legacy_at("trajectory_kind").legacy_string() != kind)
            throw std::runtime_error("trajectory kind does not match its descriptor");
        if (value.legacy_at("parent_frame").legacy_string() != convention.legacy_at("world_frame").legacy_string() ||
            value.legacy_at("child_frame").legacy_string() != convention.legacy_at("sensor_frame").legacy_string())
            throw std::runtime_error("trajectory coordinate frames disagree with the manifest");
        pose_sample sample{value.legacy_at("stamp_ns").legacy_unsigned_integer(), detail::legacy_read_pose(value)};
        if (!result.empty() && sample.stamp_ns <= result.back().stamp_ns)
            throw std::runtime_error("trajectory timestamps are not strictly increasing");
        result.push_back(sample);
    }
    if (!file.eof() || file.bad() || result.empty() || file_identity(path) != before)
        throw std::runtime_error("trajectory is empty, changed or could not be read");
    return result;
}

operation_status load_json_trajectories(dataset& input, const detail::json_value& metadata)
{
    if (input.type != "recorded_spinning_lidar" && input.type != "synthetic_spinning_lidar")
        throw std::runtime_error("unsupported spinning lidar dataset type");
    const auto& convention = metadata.legacy_at("frame_convention");
    if (metadata.members.contains("trajectory_ground_truth_file")) {
        auto ground_truth = read_trajectory(detail::legacy_dataset_path(input.root,
            metadata.legacy_at("trajectory_ground_truth_file").legacy_string()), "ground_truth", convention);
        if (!ground_truth) return std::unexpected(ground_truth.error());
        input.ground_truth = std::move(*ground_truth);
    }
    else if (input.type != "recorded_spinning_lidar")
        throw std::runtime_error("synthetic recording requires a ground truth trajectory");
    auto odometry = read_trajectory(detail::legacy_dataset_path(input.root,
        metadata.legacy_at("odometry_file").legacy_string()), "odometry", convention);
    if (!odometry) return std::unexpected(odometry.error());
     input.odometry = std::move(*odometry);
    return {};
}

void validate_trajectory_coverage(const dataset& input)
{
    if (input.frames.empty()) throw std::runtime_error("recording has no frames");
    const auto begin = input.frames.front().start_stamp_ns;
    auto minimum_stamp = begin;
    std::uint64_t end{};
    for (const auto& frame : input.frames) {
        minimum_stamp = std::min(minimum_stamp, frame.start_stamp_ns);
        end = std::max(end, frame.end_stamp_ns);
    }
    for (const auto* trajectory : {&input.ground_truth, &input.odometry})
    {
        if (trajectory->empty() && trajectory == &input.ground_truth && input.type == "recorded_spinning_lidar") continue;
        if (trajectory->empty() || trajectory->front().stamp_ns > minimum_stamp || trajectory->back().stamp_ns < end)
            throw std::runtime_error("trajectory does not cover every scan");
    }
}
}

pose sample_pose(std::span<const pose_sample> trajectory, std::uint64_t stamp_ns)
{
    if (trajectory.empty()) throw std::invalid_argument("cannot sample an empty trajectory");
    if (stamp_ns < trajectory.front().stamp_ns || stamp_ns > trajectory.back().stamp_ns)
        throw std::out_of_range("pose timestamp is outside trajectory coverage");
    const auto upper = std::lower_bound(trajectory.begin(), trajectory.end(), stamp_ns,
        [](const auto& sample, std::uint64_t stamp) { return sample.stamp_ns < stamp; });
    if (upper == trajectory.end()) throw std::invalid_argument("invalid trajectory ordering");
    validate_pose_sample(*upper);
     if (upper == trajectory.begin() || upper->stamp_ns == stamp_ns) return upper->pose;
    const auto& left = *(upper - 1);
    validate_pose_sample(left);
    if (left.stamp_ns >= upper->stamp_ns)
        throw std::invalid_argument("invalid trajectory interval");
    const double alpha = static_cast<double>(stamp_ns - left.stamp_ns) /
                         static_cast<double>(upper->stamp_ns - left.stamp_ns);
    const auto a = normalized(left.pose.rotation);
    auto b = normalized(upper->pose.rotation);
    double cosine = double(a.x) * b.x + double(a.y) * b.y + double(a.z) * b.z + double(a.w) * b.w;
    if (cosine < 0) {
        b = {-b.x, -b.y, -b.z, -b.w};
        cosine = -cosine;
    }
    double left_weight = 1 - alpha;
     double right_weight = alpha;
    if (cosine < 0.9995)
    {
        const auto angle = std::acos(std::clamp(cosine, 0.0, 1.0));
        const auto sine = std::sin(angle);
        left_weight = std::sin((1 - alpha) * angle) / sine;
        right_weight = std::sin(alpha * angle) / sine;
    }
    const auto& first = left.pose.translation;
    const auto& last = upper->pose.translation;
    pose result;
    result.translation = {static_cast<float>(std::lerp(double(first.x), double(last.x), alpha)),
                          static_cast<float>(std::lerp(double(first.y), double(last.y), alpha)),
                           static_cast<float>(std::lerp(double(first.z), double(last.z), alpha))};
    result.rotation = normalized({static_cast<float>(left_weight * a.x + right_weight * b.x),
                                  static_cast<float>(left_weight * a.y + right_weight * b.y),
                                  static_cast<float>(left_weight * a.z + right_weight * b.z),
                                  static_cast<float>(left_weight * a.w + right_weight * b.w)});
    return result;
}

std::span<const pose_sample> pose_stream(const dataset& input, pose_source source)
{
    const auto* result = source == pose_source::ground_truth ? &input.ground_truth : &input.odometry;
    if ((source != pose_source::ground_truth && source != pose_source::odometry) || result->empty())
        throw std::invalid_argument("requested trajectory is unavailable");
     return *result;
}
}

#include <charconv>

namespace covoxslam
{
namespace
{
std::size_t frame_beam_count(const frame_descriptor& frame);

std::uint64_t input_structure_hash(const dataset& input, const detail::json_value& metadata)
{
    std::vector<std::filesystem::path> paths{input.manifest_path};
    for (const auto key : {"sensor_file", "world_file", "timeline_file"})
    {
        if (metadata.members.contains(key))
            paths.push_back(detail::legacy_dataset_path(input.root, metadata.legacy_at(key).legacy_string()));
    }
    for (const auto& frame : input.frames)
    {
        if (hash_file(frame.metadata_path).content_hash != frame.metadata_content_hash)
            throw std::runtime_error("frame metadata changed after parsing");
        paths.push_back(frame.metadata_path);
    }
    cache_buffer structure;
    for (const auto& path : paths) {
        const auto relative = path.lexically_relative(input.root).generic_string();
         const auto source = hash_file(path);
        structure.append(static_cast<std::uint64_t>(relative.size()));
        structure.append(hash_bytes({reinterpret_cast<const std::uint8_t*>(relative.data()), relative.size()}));
        structure.append(source.identity.size);
        structure.append(source.content_hash);
    }
    for (const auto key : {"trajectory_ground_truth_file", "odometry_file"})
    {
        if (!metadata.members.contains(key)) continue;
        const auto identity = file_identity(detail::legacy_dataset_path(input.root, metadata.legacy_at(key).legacy_string()));
        structure.append(identity.size);
        structure.append(identity.modified);
    }
    return hash_bytes(structure.bytes);
}

void prepare_payload_index(dataset& input)
{
    input.payload_index.clear();
    for (const auto& frame : input.frames)
    {
        const auto count = frame_beam_count(frame);
        const std::array<std::pair<std::filesystem::path, std::uint64_t>, 5> payloads{{
            {frame.raw_xyz_path, count * 12ULL}, {frame.deskewed_xyz_path, count * 12ULL},
             {frame.range_path, count * 4ULL}, {frame.status_path, count},
            {frame.time_offset_ns_path, frame.columns * 4ULL}}};
        for (const auto& [path, size] : payloads) {
            if (path.empty()) continue;
            const auto resolved = detail::legacy_dataset_path(input.root, path.lexically_relative(input.root));
            const auto identity = file_identity(resolved);
            if (identity.size != size)
                throw std::runtime_error("indexed payload size disagrees with frame metadata");
            input.payload_index.emplace(resolved, payload_cache_descriptor{identity, 0});
        }
    }
}

void update_content_fingerprint(dataset& input, std::uint64_t structure_hash)
{
    auto hash = hash_basis;
    const auto append = [&hash](const auto& value) {
        hash = hash_bytes({reinterpret_cast<const std::uint8_t*>(&value), sizeof(value)}, hash);
    };
    append(structure_hash);
    for (const auto& [path, payload] : input.payload_index)
    {
        append(payload.identity.size);
         append(payload.content_hash);
    }
    for (const auto* trajectory : {&input.ground_truth, &input.odometry})
    {
        append(static_cast<std::uint64_t>(trajectory->size()));
        for (const auto& sample : *trajectory) {
            append(sample.stamp_ns);
            const auto& p = sample.pose.translation;
            const auto& q = sample.pose.rotation;
            for (const float value : {p.x, p.y, p.z, q.x, q.y, q.z, q.w}) append(value);
        }
    }
    std::array<char, 16> hexadecimal;
    const auto result = std::to_chars(hexadecimal.data(), hexadecimal.data() + hexadecimal.size(),
                                     hash, 16);
    input.content_fingerprint = "dataset-native-v1-" + std::string(hexadecimal.data(), result.ptr);
}

void index_dataset_payloads(dataset& input, std::uint64_t structure_hash)
{
    prepare_payload_index(input);
    for (auto& [path, payload] : input.payload_index)
    {
        const auto verified = hash_file(path);
        if (verified.identity != payload.identity)
            throw std::runtime_error("payload changed while the input index was built");
         payload = verified;
    }
    update_content_fingerprint(input, structure_hash);
}

void verify_frame_bytes(const dataset& input, const std::filesystem::path& path,
                         std::span<const std::uint8_t> bytes)
{
    if (input.payload_index.empty()) return;
    const auto entry = input.payload_index.find(path);
    if (entry == input.payload_index.end() || entry->second.identity.size != bytes.size() ||
        entry->second.content_hash != hash_bytes(bytes))
        throw std::runtime_error("frame payload does not match the verified input index");
}

void append_cached_trajectory(cache_buffer& buffer, std::span<const pose_sample> trajectory)
{
    if (trajectory.size() > maximum_trajectory_samples)
        throw std::length_error("trajectory cache exceeds sample limit");
     buffer.append(static_cast<std::uint64_t>(trajectory.size()));
    for (const auto& sample : trajectory) {
        validate_pose_sample(sample);
        buffer.append(sample.stamp_ns);
        const auto& p = sample.pose.translation;
        const auto& q = sample.pose.rotation;
        for (const float component : {p.x, p.y, p.z, q.x, q.y, q.z, q.w})
            buffer.append(component);
    }
}

void write_input_cache(const dataset& input, std::uint64_t structure_hash)
{
    try
    {
        cache_buffer buffer;
        buffer.append(cache_magic);
        buffer.append(structure_hash);
        buffer.append(static_cast<std::uint64_t>(input.payload_index.size()));
        for (const auto& [path, payload] : input.payload_index) {
            if (file_identity(path) != payload.identity) return;
            buffer.append(payload.identity.size);
             buffer.append(payload.identity.modified);
            buffer.append(payload.content_hash);
        }
        append_cached_trajectory(buffer, input.ground_truth);
        append_cached_trajectory(buffer, input.odometry);
        write_cache_bytes(input.root, buffer);
    }
    catch (const std::exception&)
    {}
}
}
}

namespace covoxslam
{
namespace
{
std::vector<pose_sample> read_cached_trajectory(cache_buffer& buffer)
{
    const auto count = buffer.take<std::uint64_t>();
    if (count > maximum_trajectory_samples || count > buffer.remaining() / 36)
        throw std::runtime_error("invalid trajectory cache sample count");
    std::vector<pose_sample> result;
     result.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index)
    {
        pose_sample sample;
        sample.stamp_ns = buffer.take<std::uint64_t>();
        auto& p = sample.pose.translation;
        auto& q = sample.pose.rotation;
        for (float* component : {&p.x, &p.y, &p.z, &q.x, &q.y, &q.z, &q.w})
            *component = buffer.take<float>();
        validate_pose_sample(sample);
        if (!result.empty() && sample.stamp_ns <= result.back().stamp_ns)
            throw std::runtime_error("cached trajectory timestamps are not strictly increasing");
         result.push_back(sample);
    }
    return result;
}

bool read_input_cache(dataset& input, std::uint64_t structure_hash, bool ground_truth_expected)
{
    try {
        auto buffer = read_cache_bytes(detail::legacy_dataset_path(input.root, ".covox_input_cache_v1.bin"));
        if (buffer.take<std::uint64_t>() != structure_hash ||
            buffer.take<std::uint64_t>() != input.payload_index.size()) return false;
        auto payloads = input.payload_index;
        for (auto& [path, payload] : payloads)
        {
            const cache_file_identity saved{buffer.take<std::uint64_t>(), buffer.take<std::int64_t>()};
            if (saved != payload.identity) return false;
             payload.content_hash = buffer.take<std::uint64_t>();
        }
        auto ground_truth = read_cached_trajectory(buffer);
        auto odometry = read_cached_trajectory(buffer);
        if (buffer.remaining() != 0 || odometry.empty() || ground_truth_expected == ground_truth.empty()) return false;
        for (const auto& [path, payload] : payloads)
            if (file_identity(path) != payload.identity) return false;
        input.payload_index = std::move(payloads);
        input.ground_truth = std::move(ground_truth);
        input.odometry = std::move(odometry);
        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

operation_status load_dataset_inputs(dataset& input, const detail::json_value& metadata,
                                     dataset_load_options options)
{
    if (options.verification != dataset_verification_mode::prefer_cache &&
        options.verification != dataset_verification_mode::force_full)
        throw std::invalid_argument("unsupported input verification mode");
    const auto structure_hash = input_structure_hash(input, metadata);
    prepare_payload_index(input);
    const bool cached = options.verification == dataset_verification_mode::prefer_cache &&
                        read_input_cache(input, structure_hash, metadata.members.contains("trajectory_ground_truth_file"));
    if (!cached) {
        auto loaded = load_json_trajectories(input, metadata);
        if (!loaded) return std::unexpected(loaded.error());
         index_dataset_payloads(input, structure_hash);
    }
    if (input_structure_hash(input, metadata) != structure_hash)
        throw std::runtime_error("recording metadata changed while inputs were loaded");
    validate_trajectory_coverage(input);
    input.input_source = cached ? dataset_input_source::validated_cache : dataset_input_source::full_scan;
    update_content_fingerprint(input, structure_hash);
    return {};
}
}
}
