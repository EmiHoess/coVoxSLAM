#include <covoxslam/dataset.hpp>
#include "json_reader.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace covoxslam
{
namespace detail
{
pose read_pose(const json_value& value)
{
    const auto& translation = value.at("translation_m");
    const auto& rotation = value.at("quaternion_xyzw");
    if (translation.kind != json_kind::array || translation.items.size() != 3 ||
        rotation.kind != json_kind::array || rotation.items.size() != 4)
        throw std::runtime_error("invalid pose dimensions");
     auto scalar = [](const json_value& item)
    {
        const double number = item.number();
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
std::filesystem::path frame_buffer_path(
    const detail::json_value& buffers, std::string_view name,
    std::string_view dtype, std::string_view layout,
    std::initializer_list<std::size_t> shape,
    const std::filesystem::path& root,
    std::string_view unit = {}, std::string_view encoding = {})
{
    const auto& buffer = buffers.at(name);
    const auto& shape_value = buffer.at("shape");
    const auto& dimensions = shape_value.items;
    if (shape_value.kind != detail::json_kind::array || buffer.at("dtype").string() != dtype ||
        buffer.at("layout").string() != layout ||
         buffer.at("byte_order").string() != "little_endian" ||
        dimensions.size() != shape.size())
        throw std::runtime_error("unsupported organized buffer format");
    if (!unit.empty() && buffer.at("unit").string() != unit)
        throw std::runtime_error("unsupported frame buffer unit");
    if (!encoding.empty() && buffer.members.contains("invalid_encoding") &&
        buffer.at("invalid_encoding").string() != encoding)
        throw std::runtime_error("unsupported frame zero encoding");
    std::size_t index{};
    for (const auto extent : shape) {
        if (dimensions[index++].unsigned_integer() != extent)
            throw std::runtime_error("buffer shape does not match the frame");
    }
    return detail::dataset_path(root, buffer.at("file").string());
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

frame_descriptor read_frame_descriptor(
    const std::filesystem::path& root, const std::filesystem::path& metadata_path)
{
    frame_descriptor frame;
    frame.metadata_path = detail::dataset_path(root, metadata_path.lexically_relative(root));
    const auto value = detail::read_json(frame.metadata_path);
    if (value.at("schema_version").string() != "covox.spinning_lidar.v2")
        throw std::runtime_error("unsupported frame schema");
    if (value.at("document_type").string() != "spinning_lidar_frame")
        throw std::runtime_error("unexpected frame document type");
    const auto id = value.at("frame_id").unsigned_integer();
    if (id > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("frame identifier exceeds the integer range");
     frame.frame_id = static_cast<int>(id);
    if (value.members.contains("submap_id")) {
        const double id_value = value.at("submap_id").number();
        if (id_value < -1 || id_value > std::numeric_limits<int>::max() ||
            std::trunc(id_value) != id_value)
            throw std::runtime_error("invalid submap identifier");
        frame.submap_id = static_cast<int>(id_value);
    }
    frame.start_stamp_ns = value.at("header_stamp_ns").unsigned_integer();
    frame.end_stamp_ns = value.at("scan_end_stamp_ns").unsigned_integer();
    frame.rows = value.at("organized_rows").unsigned_integer();
    frame.columns = value.at("organized_cols").unsigned_integer();
     frame.beam_count = value.at("beam_count").unsigned_integer();
    frame.hit_count = value.at("hit_count").unsigned_integer();
    frame.dropout_count = value.at("dropout_count").unsigned_integer();
    frame.invalid_count = value.at("invalid_count").unsigned_integer();
    frame.outlier_count = value.at("outlier_count").unsigned_integer();
    const auto count = frame_beam_count(frame);
    if (frame.hit_count > count || frame.dropout_count > count - frame.hit_count ||
        frame.invalid_count != count - frame.hit_count - frame.dropout_count ||
        frame.outlier_count > frame.hit_count)
        throw std::runtime_error("inconsistent frame counters");
    frame.ground_truth_start = detail::read_pose(value.at("pose_world_lidar_start"));
    frame.ground_truth_end = detail::read_pose(value.at("pose_world_lidar_end"));
    frame.odometry_start = value.members.contains("odometry_pose_world_lidar_start")
        ? detail::read_pose(value.at("odometry_pose_world_lidar_start"))
        : frame.ground_truth_start;
     frame.odometry_end = value.members.contains("odometry_pose_world_lidar_end")
        ? detail::read_pose(value.at("odometry_pose_world_lidar_end"))
        : frame.ground_truth_end;
    const auto& buffers = value.at("buffers");
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

dataset load_dataset(const std::filesystem::path& root)
{
    using namespace detail;
    dataset result;
    result.root = std::filesystem::canonical(root);
    const auto metadata = read_json(dataset_path(result.root, "dataset.json"));
    result.schema_version = metadata.at("schema_version").string();
    if (result.schema_version != "covox.spinning_lidar.v2" ||
        metadata.at("document_type").string() != "spinning_lidar_dataset")
        throw std::runtime_error("unsupported dataset schema");
    result.name = metadata.at("name").string();
     result.type = metadata.at("type").string();
    const auto& units = metadata.at("units");
    if (units.at("distance").string() != "meters" || units.at("angle").string() != "radians" ||
        units.at("timestamp").string() != "nanoseconds" || units.at("range").string() != "millimeters")
        throw std::runtime_error("unsupported dataset units");
    const auto calibration = read_json(dataset_path(result.root, metadata.at("sensor_file").string()));
    if (calibration.at("schema_version").string() != result.schema_version ||
        calibration.at("document_type").string() != "spinning_lidar_sensor")
        throw std::runtime_error("unsupported sensor schema");
    const auto& convention = metadata.at("frame_convention");
    if (convention.at("handedness").string() != "right_handed" ||
        convention.at("quaternion_order").string() != "xyzw" ||
        convention.at("transform_notation").string() != "T_parent_child_maps_child_to_parent")
        throw std::runtime_error("unsupported coordinate convention");
    const auto& timing = metadata.at("timestamp_convention");
    if (timing.at("origin").string() != "dataset_relative" ||
        timing.at("scan_header").string() != "first_column" ||
        timing.at("beam_offsets").string() != "relative_to_scan_header")
        throw std::runtime_error("unsupported timestamp convention");
    auto& sensor = result.sensor;
    sensor.model = calibration.at("sensor_model").string();
    sensor.acquisition_model = calibration.at("acquisition_model").string();
    sensor.rings = calibration.at("rings").unsigned_integer();
    sensor.columns = calibration.at("columns").unsigned_integer();
    if (!sensor.rings || !sensor.columns || sensor.rings > 16777216 / sensor.columns)
        throw std::runtime_error("invalid sensor dimensions");
    if (calibration.at("beam_count").unsigned_integer() != sensor.rings * sensor.columns)
        throw std::runtime_error("sensor beam count mismatch");
     sensor.scan_rate_hz = calibration.at("scan_rate_hz").number();
    sensor.scan_period_ns = calibration.at("scan_period_ns").unsigned_integer();
    sensor.azimuth_start_rad = calibration.at("azimuth_start_rad").number();
    sensor.azimuth_coverage_rad = calibration.at("azimuth_coverage_rad").number();
    const auto direction = calibration.at("rotation_direction").string();
    if (direction != "clockwise" && direction != "counter_clockwise")
        throw std::runtime_error("unsupported sensor rotation direction");
    sensor.counter_clockwise = direction == "counter_clockwise";
    const auto& angles = calibration.at("vertical_angles_rad");
    if (angles.kind != json_kind::array || angles.items.size() != sensor.rings)
        throw std::runtime_error("invalid vertical angle table");
    for (const auto& angle : angles.items)
    {
        const auto radians = angle.number();
         if (std::abs(radians) > 1.5707963267948966) throw std::runtime_error("invalid beam elevation");
        sensor.vertical_angles_rad.push_back(static_cast<float>(radians));
    }
    const auto minimum = calibration.at("min_range_m").number();
    const auto maximum = calibration.at("max_range_m").number();
    if (minimum < 0 || maximum <= minimum || maximum > std::numeric_limits<float>::max() ||
        sensor.scan_rate_hz <= 0 || !sensor.scan_period_ns || sensor.azimuth_coverage_rad <= 0 ||
        sensor.azimuth_coverage_rad > 6.283185307179587)
        throw std::runtime_error("invalid sensor ranges or timing");
    sensor.min_range_m = static_cast<float>(minimum);
    sensor.max_range_m = static_cast<float>(maximum);
    sensor.body_from_lidar = read_pose(calibration.at("T_parent_child"));
    const std::filesystem::path pattern(metadata.at("frame_metadata_pattern").string());
     const auto directory = dataset_path(result.root, pattern.parent_path());
    const auto filename = pattern.filename().string();
    const auto wildcard = filename.find('*');
    if (wildcard == std::string::npos || filename.find('*', wildcard + 1) != std::string::npos)
        throw std::runtime_error("expected one frame metadata wildcard");
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        const auto name = entry.path().filename().string();
        if (entry.is_regular_file() && name.starts_with(filename.substr(0, wildcard)) &&
            name.ends_with(filename.substr(wildcard + 1)))
        {
            auto frame = read_frame_descriptor(result.root,
                dataset_path(result.root, pattern.parent_path() / entry.path().filename()));
            if (frame.rows != sensor.rings || frame.columns != sensor.columns)
                throw std::runtime_error("frame dimensions differ from the sensor calibration");
            result.frames.push_back(std::move(frame));
        }
    }
     if (result.frames.size() != metadata.at("num_frames").unsigned_integer())
        throw std::runtime_error("frame metadata count mismatch");
    std::sort(result.frames.begin(), result.frames.end(), [](const auto& left, const auto& right) {
        return left.frame_id < right.frame_id;
    });
    for (std::size_t index = 1; index < result.frames.size(); ++index)
        if (result.frames[index - 1].frame_id == result.frames[index].frame_id)
            throw std::runtime_error("duplicate frame identifier");
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
    std::size_t count, std::size_t width)
{
    if (path.empty() || width == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / width)
        throw std::runtime_error("missing frame buffer or invalid byte count");
    const auto resolved = detail::dataset_path(input.root, path.lexically_relative(input.root));
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
    const dataset& input, const std::filesystem::path& path, std::size_t count)
{
    const auto bytes = read_frame_bytes(input, path, count, 4);
    std::vector<std::uint32_t> result(count);
    for (std::size_t index = 0; index < count; ++index)
        result[index] = decode_frame_u32(bytes.data() + index * 4);
    return result;
}

std::vector<vec3f> read_frame_xyz(
    const dataset& input, const std::filesystem::path& path, std::size_t count)
{
    const auto bytes = read_frame_bytes(input, path, count, 12);
     std::vector<vec3f> result(count);
    for (std::size_t index = 0; index < count; ++index) {
        const auto* data = bytes.data() + index * 12;
        const vec3f point{
            std::bit_cast<float>(decode_frame_u32(data)),
            std::bit_cast<float>(decode_frame_u32(data + 4)),
            std::bit_cast<float>(decode_frame_u32(data + 8))};
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            throw std::runtime_error("frame buffer contains a non-finite coordinate");
        result[index] = point;
    }
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
        const auto path = detail::dataset_path(
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
        frame.raw_xyz = read_frame_xyz(input, frame.descriptor.raw_xyz_path, count);
    if (options.deskewed_xyz)
        frame.deskewed_xyz = read_frame_xyz(input, frame.descriptor.deskewed_xyz_path, count);
    if (options.range)
        frame.range_mm = read_frame_u32(input, frame.descriptor.range_path, count);
    if (options.status)
        frame.status = read_frame_bytes(input, frame.descriptor.status_path, count, 1);
    if (options.time_offset_ns)
        frame.time_offset_ns = read_frame_u32(
            input, frame.descriptor.time_offset_ns_path, frame.descriptor.columns);
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
    return frame;
}
}
