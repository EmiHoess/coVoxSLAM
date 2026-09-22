#include <covoxslam/input/dataset.hpp>
#include "json_reader.hpp"
#include "dataset_metadata.hpp"
#include "dataset_format.hpp"
#include <covoxslam/core/byte_units.hpp>
#include <covoxslam/core/hash.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <chrono>
#include <cstring>
#include <limits>

namespace covoxslam
{
namespace dataset_wire = detail::dataset_format;

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
    if (!is_unit_quaternion(result.rotation))
        return std::unexpected(operation_error{"pose quaternion is not normalized"});
    result.rotation = normalized(result.rotation);
     return result;
}

}

namespace
{
operation_result<payload_cache_descriptor> hash_file(const std::filesystem::path& path);
template<class integer> operation_status assign_unsigned_member(
    const detail::json_value& object, std::string_view name, integer& destination);
operation_status expect_string_member(const detail::json_value& object, std::string_view name,
                                      std::string_view expected, std::string_view mismatch);

operation_result<std::filesystem::path> frame_buffer_path(
    const detail::json_value& buffers, std::string_view name,
    std::string_view dtype, std::string_view layout,
    std::initializer_list<std::size_t> shape,
    const std::filesystem::path& root,
    std::string_view unit = {}, std::string_view encoding = {})
{
    const auto buffer_member = buffers.member(name);
    if (!buffer_member) return std::unexpected(buffer_member.error());
    const auto& buffer = buffer_member->get();
    const auto shape_member = buffer.member("shape");
    if (!shape_member) return std::unexpected(shape_member.error());
    const auto& shape_value = shape_member->get();
    const auto& dimensions = shape_value.items;
    if (shape_value.kind != detail::json_kind::array || dimensions.size() != shape.size())
        return std::unexpected(operation_error{"unsupported organized buffer format"});
    if (const auto checked = expect_string_member(buffer, "dtype", dtype, "unsupported organized buffer format"); !checked)
        return std::unexpected(checked.error());
    if (const auto checked = expect_string_member(buffer, "layout", layout, "unsupported organized buffer format"); !checked)
        return std::unexpected(checked.error());
    if (const auto checked = expect_string_member(buffer, "byte_order", "little_endian", "unsupported organized buffer format"); !checked)
        return std::unexpected(checked.error());
    if (!unit.empty()) { const auto checked = expect_string_member(buffer, "unit", unit, "unsupported frame buffer unit");
        if (!checked) return std::unexpected(checked.error()); }
    if (!encoding.empty() && buffer.members.contains("invalid_encoding")) { const auto checked = expect_string_member(buffer, "invalid_encoding", encoding, "unsupported frame zero encoding");
        if (!checked) return std::unexpected(checked.error()); }
    std::size_t index{};
    for (const auto extent : shape) {
        const auto dimension = dimensions[index++].unsigned_integer();
        if (!dimension) return std::unexpected(dimension.error());
        if (*dimension != extent) return std::unexpected(operation_error{"buffer shape does not match the frame"});
    }
    const auto file_member = buffer.member("file");
    if (!file_member) return std::unexpected(file_member.error());
    const auto file = file_member->get().string();
    if (!file) return std::unexpected(file.error());
    return detail::dataset_path(root, file->get());
}

operation_status assign_pose_member(
    const detail::json_value& object, std::string_view name, pose& destination)
{
    const auto member = object.member(name);
    if (!member) return std::unexpected(member.error());
    const auto value = detail::read_pose(member->get());
    if (!value) return std::unexpected(value.error());
    destination = *value;
    return {};
}

operation_result<std::size_t> frame_beam_count(const frame_descriptor& frame)
{
    constexpr std::size_t maximum = 16U * 1024U * 1024U;
    if (frame.rows == 0 || frame.columns == 0 ||
        frame.rows > maximum / frame.columns)
        return std::unexpected(operation_error{"frame dimensions exceed the supported bounds"});
    const auto count = frame.rows * frame.columns;
    if (frame.beam_count != count || frame.end_stamp_ns <= frame.start_stamp_ns)
        return std::unexpected(operation_error{"invalid frame beam count or scan interval"});
     return count;
}
}

operation_result<frame_descriptor> read_frame_descriptor(
    const std::filesystem::path& root, const std::filesystem::path& metadata_path)
{
    frame_descriptor frame;
    const auto checked_path = detail::dataset_path(root, metadata_path.lexically_relative(root));
    if (!checked_path) return std::unexpected(checked_path.error());
    frame.metadata_path = *checked_path;
    const auto hash = hash_file(frame.metadata_path);
    if (!hash) return std::unexpected(hash.error());
    frame.metadata_content_hash = hash->content_hash;
    const auto parsed = detail::read_metadata(frame.metadata_path);
    if (!parsed) return std::unexpected(parsed.error());
    const auto& value = *parsed;
    if (const auto checked = expect_string_member(value, "schema_version", dataset_wire::schema, "unsupported frame schema"); !checked)
        return std::unexpected(checked.error());
    if (const auto checked = expect_string_member(value, "document_type", dataset_wire::frame_document, "unexpected frame document type"); !checked)
        return std::unexpected(checked.error());
    const auto id_member = value.member("frame_id");
    if (!id_member) return std::unexpected(id_member.error());
    const auto id = id_member->get().unsigned_integer();
    if (!id) return std::unexpected(id.error());
    if (*id > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        return std::unexpected(operation_error{"frame identifier exceeds the integer range"});
     frame.frame_id = static_cast<int>(*id);
    if (value.members.contains("submap_id")) {
        const auto submap_member = value.member("submap_id");
        if (!submap_member) return std::unexpected(submap_member.error());
        const auto id_value = submap_member->get().number();
        if (!id_value) return std::unexpected(id_value.error());
        if (*id_value < -1 || *id_value > std::numeric_limits<int>::max() ||
            std::trunc(*id_value) != *id_value)
            return std::unexpected(operation_error{"invalid submap identifier"});
        frame.submap_id = static_cast<int>(*id_value);
    }
    operation_status assigned;
    if (!(assigned = assign_unsigned_member(value, "header_stamp_ns", frame.start_stamp_ns))) return std::unexpected(assigned.error());
    if (!(assigned = assign_unsigned_member(value, "scan_end_stamp_ns", frame.end_stamp_ns))) return std::unexpected(assigned.error());
    if (!(assigned = assign_unsigned_member(value, "organized_rows", frame.rows))) return std::unexpected(assigned.error());
    if (!(assigned = assign_unsigned_member(value, "organized_cols", frame.columns))) return std::unexpected(assigned.error());
    if (!(assigned = assign_unsigned_member(value, "beam_count", frame.beam_count))) return std::unexpected(assigned.error());
    if (!(assigned = assign_unsigned_member(value, "hit_count", frame.hit_count))) return std::unexpected(assigned.error());
    if (!(assigned = assign_unsigned_member(value, "dropout_count", frame.dropout_count))) return std::unexpected(assigned.error());
    if (!(assigned = assign_unsigned_member(value, "invalid_count", frame.invalid_count))) return std::unexpected(assigned.error());
    if (!(assigned = assign_unsigned_member(value, "outlier_count", frame.outlier_count))) return std::unexpected(assigned.error());
    constexpr std::size_t maximum = 16U * 1024U * 1024U;
    if (!frame.rows || !frame.columns || frame.rows > maximum / frame.columns)
        return std::unexpected(operation_error{"frame dimensions exceed the supported bounds"});
    const auto count = frame.rows * frame.columns;
    if (frame.beam_count != count || frame.end_stamp_ns <= frame.start_stamp_ns)
        return std::unexpected(operation_error{"invalid frame beam count or scan interval"});
    if (frame.hit_count > count || frame.dropout_count > count - frame.hit_count ||
        frame.invalid_count != count - frame.hit_count - frame.dropout_count ||
        frame.outlier_count > frame.hit_count)
        return std::unexpected(operation_error{"inconsistent frame counters"});
    if (!(assigned = assign_pose_member(value, "pose_world_lidar_start", frame.ground_truth_start))) return std::unexpected(assigned.error());
    if (!(assigned = assign_pose_member(value, "pose_world_lidar_end", frame.ground_truth_end))) return std::unexpected(assigned.error());
    frame.odometry_start = frame.ground_truth_start;
    frame.odometry_end = frame.ground_truth_end;
    if (value.members.contains("odometry_pose_world_lidar_start") &&
        !(assigned = assign_pose_member(value, "odometry_pose_world_lidar_start", frame.odometry_start)))
        return std::unexpected(assigned.error());
    if (value.members.contains("odometry_pose_world_lidar_end") &&
        !(assigned = assign_pose_member(value, "odometry_pose_world_lidar_end", frame.odometry_end)))
        return std::unexpected(assigned.error());
    const auto buffers_member = value.member("buffers");
    if (!buffers_member) return std::unexpected(buffers_member.error());
    const auto& buffers = buffers_member->get();
    const auto rows = frame.rows;
    const auto columns = frame.columns;
    auto buffer_path = frame_buffer_path(
        buffers, "raw_xyz", "float32", "ring_major", {rows, columns, 3}, root, {}, "zero_xyz");
    if (!buffer_path) return std::unexpected(buffer_path.error());
    frame.raw_xyz_path = std::move(*buffer_path);
    if (buffers.members.contains("deskewed_xyz")) {
        buffer_path = frame_buffer_path(
            buffers, "deskewed_xyz", "float32", "ring_major", {rows, columns, 3}, root, {}, "zero_xyz");
        if (!buffer_path) return std::unexpected(buffer_path.error());
        frame.deskewed_xyz_path = std::move(*buffer_path);
    }
    buffer_path = frame_buffer_path(
        buffers, "range", "uint32", "ring_major", {rows, columns}, root, "millimeters", "zero");
    if (!buffer_path) return std::unexpected(buffer_path.error());
    frame.range_path = std::move(*buffer_path);
    buffer_path = frame_buffer_path(
        buffers, "status", "uint8", "ring_major", {rows, columns}, root);
    if (!buffer_path) return std::unexpected(buffer_path.error());
    frame.status_path = std::move(*buffer_path);
    buffer_path = frame_buffer_path(
        buffers, "time_offset_ns", "uint32", "one_per_column", {columns}, root, "nanoseconds");
    if (!buffer_path) return std::unexpected(buffer_path.error());
    frame.time_offset_ns_path = std::move(*buffer_path);
    return frame;
}

namespace
{
operation_status load_json_trajectories(dataset& input, const detail::json_value& metadata);
operation_status validate_trajectory_coverage(const dataset& input);
operation_status load_dataset_inputs(dataset& input, const detail::json_value& metadata,
                                     dataset_load_options options);
operation_result<std::uint64_t> input_structure_hash(
    const dataset& input, const detail::json_value& metadata);
operation_status index_dataset_payloads(dataset& input, std::uint64_t structure_hash);
operation_status write_input_cache(const dataset& input, std::uint64_t structure_hash);
operation_status verify_frame_bytes(const dataset& input, const std::filesystem::path& path,
                                    std::span<const std::uint8_t> bytes);

operation_status assign_string_member(
    const detail::json_value& object, std::string_view name, std::string& destination)
{
    const auto member = object.member(name);
    if (!member) return std::unexpected(member.error());
    const auto value = member->get().string();
    if (!value) return std::unexpected(value.error());
    destination = value->get();
    return {};
}

template<class integer>
operation_status assign_unsigned_member(
    const detail::json_value& object, std::string_view name, integer& destination)
{
    const auto member = object.member(name);
    if (!member) return std::unexpected(member.error());
    const auto value = member->get().unsigned_integer();
    if (!value) return std::unexpected(value.error());

    if (*value > std::numeric_limits<integer>::max())
        return std::unexpected(operation_error{"JSON unsigned integer exceeds destination range"});
    destination = static_cast<integer>(*value);
    return {};
}

operation_status assign_number_member(
    const detail::json_value& object, std::string_view name, double& destination)
{
    const auto member = object.member(name);
    if (!member) return std::unexpected(member.error());
    const auto value = member->get().number();
    if (!value) return std::unexpected(value.error());
    destination = *value;
    return {};
}

operation_status expect_string_member(
    const detail::json_value& object, std::string_view name,
    std::string_view expected, std::string_view mismatch)
{
    const auto member = object.member(name);
    if (!member) return std::unexpected(member.error());
    const auto value = member->get().string();
    if (!value) return std::unexpected(value.error());

    if (value->get() != expected) return std::unexpected(operation_error{std::string(mismatch)});
    return {};
}
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
        const bool json_exists = std::filesystem::exists(selected / dataset_wire::json_manifest, path_error);
        if (path_error) return std::unexpected(operation_error{"cannot select dataset manifest"});
        selected /= json_exists ? dataset_wire::json_manifest : dataset_wire::toml_manifest;
    }
    result.manifest_path = selected;
    auto parsed = read_metadata(selected);
    if (!parsed) return std::unexpected(parsed.error());
    const auto& metadata = *parsed;
    const auto manifest_source = hash_file(selected);
    if (!manifest_source) return std::unexpected(manifest_source.error());
    const auto manifest_hash = manifest_source->content_hash;
    const auto encoding = selected.extension() == dataset_wire::toml_extension ?
        dataset_wire::toml_encoding : dataset_wire::json_encoding;
    if (metadata.members.contains("metadata_format")) {
        const auto format = expect_string_member(metadata, "metadata_format", encoding,
                                                  "metadata format disagrees with extension");
        if (!format) return std::unexpected(format.error());
    }
    if (const auto schema = assign_string_member(metadata, "schema_version", result.schema_version); !schema)
        return std::unexpected(schema.error());
    if (result.schema_version != dataset_wire::schema)
        return std::unexpected(operation_error{"unsupported dataset schema"});
    if (const auto document = expect_string_member(metadata, "document_type", dataset_wire::dataset_document,
                                                    "unsupported dataset schema"); !document)
        return std::unexpected(document.error());
    if (const auto name = assign_string_member(metadata, "name", result.name); !name)
        return std::unexpected(name.error());
     if (const auto type = assign_string_member(metadata, "type", result.type); !type)
        return std::unexpected(type.error());
    const auto units_member = metadata.member("units");
    if (!units_member) return std::unexpected(units_member.error());
    const auto& units = units_member->get();
    for (const auto [name, expected] : {std::pair{"distance", "meters"}, std::pair{"angle", "radians"},
                                       std::pair{"timestamp", "nanoseconds"}, std::pair{"range", "millimeters"}})
    {
        const auto unit = expect_string_member(units, name, expected, "unsupported dataset units");
        if (!unit) return std::unexpected(unit.error());
    }
    std::string sensor_file;
    if (const auto file = assign_string_member(metadata, "sensor_file", sensor_file); !file)
        return std::unexpected(file.error());
    const auto checked_sensor_path = dataset_path(result.root, sensor_file);
    if (!checked_sensor_path) return std::unexpected(checked_sensor_path.error());
    const auto sensor_path = *checked_sensor_path;
    const auto sensor_source = hash_file(sensor_path);
    if (!sensor_source) return std::unexpected(sensor_source.error());
    const auto sensor_hash = sensor_source->content_hash;
    const auto parsed_calibration = read_metadata(sensor_path);
    if (!parsed_calibration) return std::unexpected(parsed_calibration.error());
    const auto& calibration = *parsed_calibration;
    if (const auto schema = expect_string_member(calibration, "schema_version", result.schema_version,
                                                  "unsupported sensor schema"); !schema)
        return std::unexpected(schema.error());
    if (const auto document = expect_string_member(calibration, "document_type", dataset_wire::sensor_document,
                                                    "unsupported sensor schema"); !document)
        return std::unexpected(document.error());
    const auto convention_member = metadata.member("frame_convention");
    if (!convention_member) return std::unexpected(convention_member.error());
    const auto& convention = convention_member->get();
    for (const auto [name, expected] : {
        std::pair{"handedness", "right_handed"}, std::pair{"quaternion_order", "xyzw"},
        std::pair{"transform_notation", "T_parent_child_maps_child_to_parent"}})
    {
        const auto checked = expect_string_member(
            convention, name, expected, "unsupported coordinate convention");
        if (!checked) return std::unexpected(checked.error());
    }
    const auto timing_member = metadata.member("timestamp_convention");
    if (!timing_member) return std::unexpected(timing_member.error());
    const auto& timing = timing_member->get();
    for (const auto [name, expected] : {
        std::pair{"origin", "dataset_relative"}, std::pair{"scan_header", "first_column"},
        std::pair{"beam_offsets", "relative_to_scan_header"}})
    {
        const auto checked = expect_string_member(
            timing, name, expected, "unsupported timestamp convention");
        if (!checked) return std::unexpected(checked.error());
    }
    auto& sensor = result.sensor;
    if (const auto model = assign_string_member(calibration, "sensor_model", sensor.model); !model)
        return std::unexpected(model.error());
    if (const auto model = assign_string_member(
        calibration, "acquisition_model", sensor.acquisition_model); !model)
        return std::unexpected(model.error());
    if (const auto rings = assign_unsigned_member(calibration, "rings", sensor.rings); !rings)
        return std::unexpected(rings.error());
    if (const auto columns = assign_unsigned_member(calibration, "columns", sensor.columns); !columns)
        return std::unexpected(columns.error());
    if (!sensor.rings || !sensor.columns || sensor.rings > 16777216 / sensor.columns)
        return std::unexpected(operation_error{"invalid sensor dimensions"});
    std::uint64_t beam_count{};
    if (const auto count = assign_unsigned_member(calibration, "beam_count", beam_count); !count)
        return std::unexpected(count.error());
    if (beam_count != sensor.rings * sensor.columns)
        return std::unexpected(operation_error{"sensor beam count mismatch"});
     if (const auto rate = assign_number_member(calibration, "scan_rate_hz", sensor.scan_rate_hz); !rate)
        return std::unexpected(rate.error());
    if (const auto period = assign_unsigned_member(
        calibration, "scan_period_ns", sensor.scan_period_ns); !period)
        return std::unexpected(period.error());
    if (const auto azimuth = assign_number_member(
        calibration, "azimuth_start_rad", sensor.azimuth_start_rad); !azimuth)
        return std::unexpected(azimuth.error());
    if (const auto coverage = assign_number_member(
        calibration, "azimuth_coverage_rad", sensor.azimuth_coverage_rad); !coverage)
        return std::unexpected(coverage.error());
    std::string direction;
    if (const auto rotation = assign_string_member(calibration, "rotation_direction", direction); !rotation)
        return std::unexpected(rotation.error());
    if (direction != "clockwise" && direction != "counter_clockwise")
        return std::unexpected(operation_error{"unsupported sensor rotation direction"});
    sensor.counter_clockwise = direction == "counter_clockwise";
    const auto angles_member = calibration.member("vertical_angles_rad");
    if (!angles_member) return std::unexpected(angles_member.error());
    const auto& angles = angles_member->get();
    if (angles.kind != json_kind::array || angles.items.size() != sensor.rings)
        return std::unexpected(operation_error{"invalid vertical angle table"});
    for (const auto& angle : angles.items)
    {
        const auto radians = angle.number();
        if (!radians) return std::unexpected(radians.error());
         if (std::abs(*radians) > 1.5707963267948966)
            return std::unexpected(operation_error{"invalid beam elevation"});
        sensor.vertical_angles_rad.push_back(static_cast<float>(*radians));
    }
    double minimum{};
    double maximum{};
    if (const auto range = assign_number_member(calibration, "min_range_m", minimum); !range)
        return std::unexpected(range.error());
    if (const auto range = assign_number_member(calibration, "max_range_m", maximum); !range)
        return std::unexpected(range.error());
    if (minimum < 0 || maximum <= minimum || maximum > std::numeric_limits<float>::max() ||
        sensor.scan_rate_hz <= 0 || !sensor.scan_period_ns || sensor.azimuth_coverage_rad <= 0 ||
        sensor.azimuth_coverage_rad > 6.283185307179587)
        return std::unexpected(operation_error{"invalid sensor ranges or timing"});
    sensor.min_range_m = static_cast<float>(minimum);
    sensor.max_range_m = static_cast<float>(maximum);
    const auto transform_member = calibration.member("T_parent_child");
    if (!transform_member) return std::unexpected(transform_member.error());
    const auto body_from_lidar = read_pose(transform_member->get());
    if (!body_from_lidar) return std::unexpected(body_from_lidar.error());
    sensor.body_from_lidar = *body_from_lidar;
    std::string frame_pattern;
    if (const auto pattern = assign_string_member(metadata, "frame_metadata_pattern", frame_pattern); !pattern)
        return std::unexpected(pattern.error());
    const std::filesystem::path pattern(frame_pattern);
     const auto checked_frame_directory = dataset_path(result.root, pattern.parent_path());
    if (!checked_frame_directory) return std::unexpected(checked_frame_directory.error());
    const auto frame_directory = *checked_frame_directory;
    const auto filename = pattern.filename().string();
    const auto wildcard = filename.find('*');
    if (wildcard == std::string::npos || filename.find('*', wildcard + 1) != std::string::npos)
        return std::unexpected(operation_error{"expected one frame metadata wildcard"});
    path_error.clear();
    std::filesystem::directory_iterator entry(frame_directory, path_error);
    const std::filesystem::directory_iterator end;
    if (path_error) return std::unexpected(operation_error{
        "cannot inspect frame metadata directory: " + path_error.message()});
    while (entry != end)
    {
        const auto name = entry->path().filename().string();
        const auto regular = entry->is_regular_file(path_error);
        if (path_error) return std::unexpected(operation_error{
            "cannot inspect frame metadata entry: " + path_error.message()});
        if (regular && name.starts_with(filename.substr(0, wildcard)) &&
            name.ends_with(filename.substr(wildcard + 1)))
        {
            const auto frame_path = dataset_path(
                result.root, pattern.parent_path() / entry->path().filename());
            if (!frame_path) return std::unexpected(frame_path.error());
            auto frame = read_frame_descriptor(result.root, *frame_path);
            if (!frame) return std::unexpected(frame.error());
            if (frame->rows != sensor.rings || frame->columns != sensor.columns)
                return std::unexpected(operation_error{"frame dimensions differ from the sensor calibration"});
            result.frames.push_back(std::move(*frame));
        }
        entry.increment(path_error);
        if (path_error) return std::unexpected(operation_error{
            "cannot enumerate frame metadata directory: " + path_error.message()});
    }
    std::uint64_t frame_count{};
    if (const auto count = assign_unsigned_member(metadata, "num_frames", frame_count); !count)
        return std::unexpected(count.error());
     if (result.frames.size() != frame_count)
        return std::unexpected(operation_error{"frame metadata count mismatch"});
    std::sort(result.frames.begin(), result.frames.end(), [](const auto& left, const auto& right) {
        return left.frame_id < right.frame_id;
    });
    for (std::size_t index = 1; index < result.frames.size(); ++index)
        if (result.frames[index - 1].frame_id == result.frames[index].frame_id)
            return std::unexpected(operation_error{"duplicate frame identifier"});
    auto inputs = load_dataset_inputs(result, metadata, options);
    if (!inputs) return std::unexpected(inputs.error());
    const auto final_manifest = hash_file(selected);
    if (!final_manifest) return std::unexpected(final_manifest.error());
    const auto final_sensor = hash_file(sensor_path);
    if (!final_sensor) return std::unexpected(final_sensor.error());
    if (manifest_hash != final_manifest->content_hash ||
        sensor_hash != final_sensor->content_hash)
        return std::unexpected(operation_error{"recording metadata changed while inputs were loaded"});
    if (options.update_cache && result.input_source == dataset_input_source::full_scan) {
        const auto structure_hash = input_structure_hash(result, metadata);
        if (!structure_hash) return std::unexpected(structure_hash.error());
        static_cast<void>(write_input_cache(result, *structure_hash));
    }
    return result;
}
}

#include <bit>
#include <fstream>

namespace covoxslam
{
namespace
{
operation_result<std::vector<std::uint8_t>> read_frame_bytes(
    const dataset& input, const std::filesystem::path& path,
    std::size_t count, std::size_t width, input_timing& timing)
{
    const auto started = std::chrono::steady_clock::now();
    if (path.empty() || width == 0 ||
        count > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / width)
        return std::unexpected(operation_error{"missing frame buffer or invalid byte count"});
    const auto resolved = detail::dataset_path(input.root, path.lexically_relative(input.root));
    if (!resolved) return std::unexpected(resolved.error());
    const auto bytes = count * width;
     std::ifstream file(*resolved, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(bytes))
        return std::unexpected(operation_error{"frame buffer byte count does not match its shape"});
    file.seekg(0);
    std::vector<std::uint8_t> result(bytes);
    file.read(reinterpret_cast<char*>(result.data()), static_cast<std::streamsize>(bytes));
    if (!file || file.gcount() != static_cast<std::streamsize>(bytes) ||
        file.peek() != std::char_traits<char>::eof() || file.bad())
        return std::unexpected(operation_error{"frame buffer could not be read completely"});
    const auto loaded = std::chrono::steady_clock::now();
    timing.read_ms += std::chrono::duration<double, std::milli>(loaded - started).count();
    timing.payload_bytes += bytes;
    const auto verified = verify_frame_bytes(input, *resolved, result);
    if (!verified) return std::unexpected(verified.error());
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

operation_result<std::vector<std::uint32_t>> read_frame_u32(
    const dataset& input, const std::filesystem::path& path, std::size_t count, input_timing& timing)
{
    auto bytes = read_frame_bytes(input, path, count, dataset_wire::uint32_width, timing);
    if (!bytes) return std::unexpected(bytes.error());
    const auto started = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> result(count);
        if (!bytes->empty()) std::memcpy(result.data(), bytes->data(), bytes->size());
    timing.decode_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}

operation_result<std::vector<vec3f>> read_frame_xyz(
    const dataset& input, const std::filesystem::path& path, std::size_t count, input_timing& timing)
{
    auto bytes = read_frame_bytes(input, path, count, dataset_wire::xyz_width, timing);
    if (!bytes) return std::unexpected(bytes.error());
    const auto started = std::chrono::steady_clock::now();
     std::vector<vec3f> result(count);
    constexpr bool native = sizeof(vec3f) == dataset_wire::xyz_width &&
        std::is_trivially_copyable_v<vec3f> && offsetof(vec3f, y) == dataset_wire::float32_width &&
        offsetof(vec3f, z) == 2 * dataset_wire::float32_width;
    if constexpr (native)
        if (!bytes->empty()) std::memcpy(result.data(), bytes->data(), bytes->size());

    for (std::size_t index = 0; index < count; ++index) {
        const auto* data = bytes->data() + index * dataset_wire::xyz_width;
        const vec3f point = native ? result[index] : vec3f{
            std::bit_cast<float>(decode_frame_u32(data)),
            std::bit_cast<float>(decode_frame_u32(data + dataset_wire::float32_width)),
            std::bit_cast<float>(decode_frame_u32(data + 2 * dataset_wire::float32_width))};
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            return std::unexpected(operation_error{"frame buffer contains a non-finite coordinate"});
        result[index] = point;
    }
    timing.decode_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    return result;
}
}

operation_result<std::size_t> organized_frame::flat_index(
    std::size_t ring, std::size_t column) const
{
    constexpr std::size_t maximum = 16U * 1024U * 1024U;
    if (!descriptor.rows || !descriptor.columns || descriptor.rows > maximum / descriptor.columns)
        return std::unexpected(operation_error{"frame dimensions exceed the supported bounds"});
    if (descriptor.beam_count != descriptor.rows * descriptor.columns ||
        descriptor.end_stamp_ns <= descriptor.start_stamp_ns)
        return std::unexpected(operation_error{"invalid frame beam count or scan interval"});
    if (ring >= descriptor.rows || column >= descriptor.columns)
        return std::unexpected(operation_error{"organized frame index is out of range"});
     return ring * descriptor.columns + column;
}

operation_result<std::reference_wrapper<const vec3f>> organized_frame::point(
    point_source source, std::size_t ring, std::size_t column) const
{
    const auto index = flat_index(ring, column);
    if (!index) return std::unexpected(index.error());
    const std::vector<vec3f>* points{};
    if (source == point_source::raw) points = &raw_xyz;
    else if (source == point_source::deskewed) points = &deskewed_xyz;
    else return std::unexpected(operation_error{"unsupported point source"});
    if (*index >= points->size())
        return std::unexpected(operation_error{"requested point source is unavailable"});
    return std::cref((*points)[*index]);
}

operation_result<std::uint64_t> organized_frame::beam_stamp_ns(std::size_t column) const
{
    if (column >= descriptor.columns)
        return std::unexpected(operation_error{"organized frame column is out of range"});
    if (column >= time_offset_ns.size())
        return std::unexpected(operation_error{"frame time offset buffer is unavailable"});
    const auto offset = time_offset_ns[column];
     if (descriptor.end_stamp_ns <= descriptor.start_stamp_ns ||
        offset >= descriptor.end_stamp_ns - descriptor.start_stamp_ns)
        return std::unexpected(operation_error{"column time offset is outside the scan interval"});
    return descriptor.start_stamp_ns + offset;
}

operation_result<bool> organized_frame::valid_return(std::size_t ring, std::size_t column) const {
    const auto index = flat_index(ring, column);
    if (!index) return std::unexpected(index.error());
    if (*index >= status.size())
        return std::unexpected(operation_error{"frame status buffer is unavailable"});
    const auto value = status[*index];
    if (value > static_cast<std::uint8_t>(beam_status::valid_outlier))
        return std::unexpected(operation_error{"frame buffer contains an unknown status code"});
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

operation_result<std::size_t> validate_frame_payload_bounds(
    const dataset& input, const frame_descriptor& frame, const frame_load_options& options)
{
    if (options.max_payload_bytes == 0)
        return std::unexpected(operation_error{"frame payload byte limit must be positive"});
    constexpr std::size_t maximum = 16U * 1024U * 1024U;
    if (!frame.rows || !frame.columns || frame.rows > maximum / frame.columns)
        return std::unexpected(operation_error{"frame dimensions exceed the supported bounds"});
    const auto count = frame.rows * frame.columns;
    if (frame.beam_count != count || frame.end_stamp_ns <= frame.start_stamp_ns)
        return std::unexpected(operation_error{"invalid frame beam count or scan interval"});
    const std::array<frame_buffer_request, 5> requests{{
        {&frame.raw_xyz_path, count, dataset_wire::xyz_width, options.raw_xyz},
        {&frame.deskewed_xyz_path, count, dataset_wire::xyz_width, options.deskewed_xyz},
        {&frame.range_path, count, dataset_wire::range_width, options.range},
         {&frame.status_path, count, dataset_wire::status_width, options.status},
        {&frame.time_offset_ns_path, frame.columns, dataset_wire::time_offset_width, options.time_offset_ns}}};
    std::size_t total{};
    for (const auto& request : requests)
    {
        if (!request.requested)
            continue;
        if (request.count > (options.max_payload_bytes - total) / request.width)
            return std::unexpected(operation_error{"requested frame buffers exceed the payload byte limit"});
        const auto bytes = request.count * request.width;
        total += bytes;
        if (request.path->empty())
            return std::unexpected(operation_error{"a requested frame buffer is absent"});
        const auto path = detail::dataset_path(
            input.root, request.path->lexically_relative(input.root));
        if (!path) return std::unexpected(path.error());
        std::error_code error;
        const auto regular = std::filesystem::is_regular_file(*path, error);
        if (error || !regular)
            return std::unexpected(operation_error{"requested frame buffer has an invalid path or size"});
        const auto file_bytes = std::filesystem::file_size(*path, error);
        if (error || file_bytes != bytes)
            return std::unexpected(operation_error{"requested frame buffer has an invalid path or size"});
    }
    return count;
}

operation_status validate_frame_contents(const organized_frame& frame)
{
    if (frame.status.empty())
        return {};
    std::array<std::size_t, 4> counts{};
    const auto valid_point = [](vec3f point) {
        return point.x != 0 || point.y != 0 || point.z != 0;
    };
    for (std::size_t index = 0; index < frame.status.size(); ++index)
    {
        const auto status = frame.status[index];
        if (status >= counts.size())
            return std::unexpected(operation_error{"frame buffer contains an unknown status code"});
        ++counts[status];
        const bool valid = status == static_cast<std::uint8_t>(beam_status::valid_hit) ||
             status == static_cast<std::uint8_t>(beam_status::valid_outlier);
        if ((!frame.range_mm.empty() && (frame.range_mm[index] != 0) != valid) ||
            (!frame.raw_xyz.empty() && valid_point(frame.raw_xyz[index]) != valid) ||
            (!frame.deskewed_xyz.empty() && valid_point(frame.deskewed_xyz[index]) != valid))
            return std::unexpected(operation_error{"frame status disagrees with zero return encoding"});
    }
    const auto& descriptor = frame.descriptor;
    if (counts[0] != descriptor.invalid_count || counts[2] != descriptor.dropout_count ||
        counts[1] + counts[3] != descriptor.hit_count || counts[3] != descriptor.outlier_count)
        return std::unexpected(operation_error{"binary status counts disagree with the frame metadata"});
    return {};
}
}

operation_result<organized_frame> load_frame(
    const dataset& input, std::size_t frame_index, frame_load_options options)
{
    if (frame_index >= input.frames.size())
        return std::unexpected(operation_error{"frame index is out of range"});
    organized_frame frame;
    frame.descriptor = input.frames[frame_index];
    const auto count = validate_frame_payload_bounds(input, frame.descriptor, options);
    if (!count) return std::unexpected(count.error());
    if (options.deskewed_xyz && frame.descriptor.deskewed_xyz_path.empty())
        return std::unexpected(operation_error{"the frame has no deskewed XYZ buffer"});
     if (options.raw_xyz) {
        auto loaded = read_frame_xyz(input, frame.descriptor.raw_xyz_path, *count, frame.timing);
        if (!loaded) return std::unexpected(loaded.error());
        frame.raw_xyz = std::move(*loaded);
    }
    if (options.deskewed_xyz) {
        auto loaded = read_frame_xyz(input, frame.descriptor.deskewed_xyz_path, *count, frame.timing);
        if (!loaded) return std::unexpected(loaded.error());
        frame.deskewed_xyz = std::move(*loaded);
    }
    if (options.range) {
        auto loaded = read_frame_u32(input, frame.descriptor.range_path, *count, frame.timing);
        if (!loaded) return std::unexpected(loaded.error());
        frame.range_mm = std::move(*loaded);
    }
    if (options.status) {
        auto loaded = read_frame_bytes(
            input, frame.descriptor.status_path, *count, dataset_wire::status_width, frame.timing);
        if (!loaded) return std::unexpected(loaded.error());
        frame.status = std::move(*loaded);
    }
    if (options.time_offset_ns) {
        auto loaded = read_frame_u32(
            input, frame.descriptor.time_offset_ns_path, frame.descriptor.columns, frame.timing);
        if (!loaded) return std::unexpected(loaded.error());
        frame.time_offset_ns = std::move(*loaded);
    }
    const auto validation_started = std::chrono::steady_clock::now();
    for (std::size_t column = 0; column < frame.time_offset_ns.size(); ++column)
    {
        const auto offset = frame.time_offset_ns[column];
        if (frame.descriptor.end_stamp_ns <= frame.descriptor.start_stamp_ns ||
            offset >= frame.descriptor.end_stamp_ns - frame.descriptor.start_stamp_ns)
            return std::unexpected(operation_error{"column time offset is outside the scan interval"});
        if (column != 0 && frame.time_offset_ns[column] <= frame.time_offset_ns[column - 1])
            return std::unexpected(operation_error{"column time offsets are not strictly increasing"});
    }
    for (const auto value : frame.status) {
        if (value > static_cast<std::uint8_t>(beam_status::valid_outlier))
            return std::unexpected(operation_error{"frame buffer contains an unknown status code"});
    }
    const auto valid = validate_frame_contents(frame);
    if (!valid) return std::unexpected(valid.error());
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
constexpr std::uint64_t hash_basis = fnv1a_offset_basis;
constexpr std::size_t maximum_cache_bytes = bytes_per_gibibyte;
constexpr std::size_t maximum_trajectory_line_bytes = bytes_per_mebibyte;
 constexpr std::size_t maximum_trajectory_samples = 16U * 1024U * 1024U;

std::uint64_t hash_bytes(std::span<const std::uint8_t> bytes, std::uint64_t hash = hash_basis)
{
    return fnv1a(std::as_bytes(bytes), hash);
}

operation_result<cache_file_identity> file_identity(const std::filesystem::path& path)
{
    std::error_code error;
    const auto regular = std::filesystem::is_regular_file(path, error);
    if (error || !regular)
        return std::unexpected(operation_error{"recording input is not a regular file"});
    const auto modified = std::filesystem::last_write_time(path, error);
    if (error)
        return std::unexpected(operation_error{"recording input could not be inspected"});
    const auto size = std::filesystem::file_size(path, error);
    if (error)
        return std::unexpected(operation_error{"recording input could not be inspected"});
    return cache_file_identity{size, static_cast<std::int64_t>(modified.time_since_epoch().count())};
}

operation_result<payload_cache_descriptor> hash_file(const std::filesystem::path& path)
{
    const auto before = file_identity(path);
    if (!before) return std::unexpected(before.error());
    std::ifstream file(path, std::ios::binary);
    if (!file) return std::unexpected(operation_error{"cannot open recording input"});
    std::array<std::uint8_t, dataset_wire::hash_chunk_bytes> bytes;
    std::uint64_t hash = hash_basis, count{};
    while (file) {
        file.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        const auto size = static_cast<std::size_t>(file.gcount());
        hash = hash_bytes(std::span(bytes).first(size), hash); count += size;
    }
    const auto after = file_identity(path);
    if (!file.eof() || file.bad() || count != before->size || !after || *after != *before)
        return std::unexpected(operation_error{"recording input changed or could not be read"});
    return payload_cache_descriptor{*before, hash};
}

class cache_buffer
{
    std::size_t position{};
public:
    std::vector<std::uint8_t> bytes;
    template<class value_type>
    operation_status append(value_type value)
    {
        if (bytes.size() > maximum_cache_bytes - sizeof(value_type))
            return std::unexpected(operation_error{"input cache exceeds byte limit"});
        const auto* first = reinterpret_cast<const std::uint8_t*>(&value);
         bytes.insert(bytes.end(), first, first + sizeof(value_type));
        return {};
    }
    template<class value_type>
    operation_result<value_type> take() {
        if (position > bytes.size() || sizeof(value_type) > bytes.size() - position)
            return std::unexpected(operation_error{"truncated input cache"});
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

operation_result<cache_buffer> read_cache_bytes(const std::filesystem::path& path)
{
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size < dataset_wire::cache_minimum_prefix_bytes || size > maximum_cache_bytes)
        return std::unexpected(operation_error{"invalid input cache length"});
    cache_buffer result;
     result.bytes.resize(static_cast<std::size_t>(size));
    std::ifstream file(path, std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(result.bytes.data()), static_cast<std::streamsize>(size)) ||
        file.peek() != std::char_traits<char>::eof() || file.bad())
        return std::unexpected(operation_error{"cannot read complete input cache"});
    std::uint64_t checksum;
    std::memcpy(&checksum, result.bytes.data() + size - 8, 8);
    result.bytes.resize(static_cast<std::size_t>(size) - 8);
    if (hash_bytes(result.bytes) != checksum)
        return std::unexpected(operation_error{"input cache checksum or native format mismatch"});
    const auto magic = result.take<std::uint64_t>();
    if (!magic || *magic != dataset_wire::cache_magic)
        return std::unexpected(operation_error{"input cache checksum or native format mismatch"});
    return result;
}

operation_status write_cache_bytes(const std::filesystem::path& root, cache_buffer& buffer)
{
    const auto checksum = buffer.append(hash_bytes(buffer.bytes));
    if (!checksum) return std::unexpected(checksum.error());
    const auto directory = detail::dataset_path(root, dataset_wire::cache_staging_directory);
    if (!directory) return std::unexpected(directory.error());
    const auto destination = detail::dataset_path(root, dataset_wire::cache_destination_filename);
    if (!destination) return std::unexpected(destination.error());
    std::error_code error;
    if (!std::filesystem::create_directory(*directory, error))
        return std::unexpected(operation_error{"cannot create input cache staging directory"});
    const auto temporary = *directory / dataset_wire::cache_staging_filename;
    operation_status result;
    std::ofstream file(temporary, std::ios::binary);
    if (!file) result = std::unexpected(operation_error{"cannot open input cache staging file"});
    else {
        file.write(reinterpret_cast<const char*>(buffer.bytes.data()),
                    static_cast<std::streamsize>(buffer.bytes.size()));
        file.close();
        if (!file) result = std::unexpected(operation_error{"cannot write complete input cache"});
        else {
#ifdef _WIN32
            if (!MoveFileExW(temporary.c_str(), destination->c_str(),
                             MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
                result = std::unexpected(operation_error{"cannot replace input cache"});
#else
            error.clear();
             std::filesystem::rename(temporary, *destination, error);
            if (error) result = std::unexpected(operation_error{"cannot replace input cache"});
#endif
        }
    }
    error.clear();
    std::filesystem::remove(temporary, error);
    error.clear();
    std::filesystem::remove(*directory, error);
    return result;
}
}
}

namespace covoxslam
{
namespace
{
operation_status validate_pose_sample(const pose_sample& sample)
{
    const auto& position = sample.pose.translation;
    const auto& rotation = sample.pose.rotation;
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z) ||
         !is_unit_quaternion(rotation))
        return std::unexpected(operation_error{"invalid trajectory pose"});
    return {};
}

operation_error trajectory_record_error(
    const std::filesystem::path& path, std::size_t line, operation_error error)
{
    error.message = path.string() + ": line " + std::to_string(line) + ": " + error.message;
    return error;
}

operation_result<std::vector<pose_sample>> read_trajectory(
    const std::filesystem::path& path, std::string_view kind,
    std::string_view parent_frame, std::string_view child_frame)
{
    const auto before = file_identity(path);
    if (!before) return std::unexpected(before.error());
    std::ifstream file(path, std::ios::binary);
    if (!file) return std::unexpected(operation_error{"cannot open trajectory"});
    std::vector<pose_sample> result;
    std::vector<char> line(maximum_trajectory_line_bytes + 2);
    std::size_t line_number{};
    while (file.getline(line.data(), static_cast<std::streamsize>(line.size())))
    {
        ++line_number;
        const auto count = static_cast<std::size_t>(file.gcount()) - (file.eof() ? 0 : 1);
        if (count == 0 || (count == 1 && line[0] == '\r')) continue;
        if (count > maximum_trajectory_line_bytes || result.size() >= maximum_trajectory_samples)
            return std::unexpected(trajectory_record_error(path, line_number,
                operation_error{"trajectory exceeds its line or sample limit"}));
        auto parsed = detail::parse_json(std::string_view(line.data(), count));
        if (!parsed) return std::unexpected(trajectory_record_error(path, line_number, parsed.error()));
        const auto& value = *parsed;
        const auto schema = expect_string_member(value, "schema_version", dataset_wire::schema,
                                                  "unsupported trajectory record schema");
        if (!schema) return std::unexpected(trajectory_record_error(path, line_number, schema.error()));
        const auto document = expect_string_member(value, "document_type", dataset_wire::trajectory_record_document,
                                                    "unsupported trajectory record schema");
        if (!document) return std::unexpected(trajectory_record_error(path, line_number, document.error()));
        const auto checked_kind = expect_string_member(
            value, "trajectory_kind", kind, "trajectory kind does not match its descriptor");
        if (!checked_kind)
            return std::unexpected(trajectory_record_error(path, line_number, checked_kind.error()));
        const auto checked_parent = expect_string_member(
            value, "parent_frame", parent_frame,
            "trajectory coordinate frames disagree with the manifest");
        if (!checked_parent)
            return std::unexpected(trajectory_record_error(path, line_number, checked_parent.error()));
        const auto checked_child = expect_string_member(
            value, "child_frame", child_frame,
            "trajectory coordinate frames disagree with the manifest");
        if (!checked_child)
            return std::unexpected(trajectory_record_error(path, line_number, checked_child.error()));
        const auto stamp_member = value.member("stamp_ns");
        if (!stamp_member)
            return std::unexpected(trajectory_record_error(path, line_number, stamp_member.error()));
        const auto stamp = stamp_member->get().unsigned_integer();
        if (!stamp) return std::unexpected(trajectory_record_error(path, line_number, stamp.error()));
        const auto pose = detail::read_pose(value);
        if (!pose) return std::unexpected(trajectory_record_error(path, line_number, pose.error()));
        pose_sample sample{*stamp, *pose};
        if (!result.empty() && sample.stamp_ns <= result.back().stamp_ns)
            return std::unexpected(trajectory_record_error(path, line_number,
                operation_error{"trajectory timestamps are not strictly increasing"}));
        result.push_back(sample);
    }
    const auto after = file_identity(path);
    if (!file.eof() || file.bad() || result.empty() || !after || *after != *before)
        return std::unexpected(operation_error{"trajectory is empty, changed or could not be read"});
    return result;
}

operation_status load_json_trajectories(dataset& input, const detail::json_value& metadata)
{
    if (input.type != dataset_wire::recorded_type && input.type != dataset_wire::synthetic_type)
        return std::unexpected(operation_error{"unsupported spinning lidar dataset type"});
    const auto convention_member = metadata.member("frame_convention");
    if (!convention_member) return std::unexpected(convention_member.error());
    const auto& convention = convention_member->get();
    std::string world_frame;
    std::string sensor_frame;
    if (const auto world = assign_string_member(convention, "world_frame", world_frame); !world)
        return std::unexpected(world.error());
    if (const auto sensor = assign_string_member(convention, "sensor_frame", sensor_frame); !sensor)
        return std::unexpected(sensor.error());

    std::vector<pose_sample> ground_truth;
    if (metadata.members.contains("trajectory_ground_truth_file"))
    {
        std::string relative;
        if (const auto file = assign_string_member(
            metadata, "trajectory_ground_truth_file", relative); !file)
            return std::unexpected(file.error());
        const auto path = detail::dataset_path(input.root, relative);
        if (!path) return std::unexpected(path.error());
        auto loaded = read_trajectory(*path, "ground_truth", world_frame, sensor_frame);
        if (!loaded) return std::unexpected(loaded.error());
        ground_truth = std::move(*loaded);
    }
    else if (input.type != dataset_wire::recorded_type)
        return std::unexpected(operation_error{"synthetic recording requires a ground truth trajectory"});

    std::string relative_odometry;
    if (const auto file = assign_string_member(metadata, "odometry_file", relative_odometry); !file)
        return std::unexpected(file.error());
    const auto odometry_path = detail::dataset_path(input.root, relative_odometry);
    if (!odometry_path) return std::unexpected(odometry_path.error());
    auto odometry = read_trajectory(*odometry_path, "odometry", world_frame, sensor_frame);
    if (!odometry) return std::unexpected(odometry.error());
    input.ground_truth = std::move(ground_truth);
    input.odometry = std::move(*odometry);
    return {};
}

operation_status validate_trajectory_coverage(const dataset& input)
{
    if (input.frames.empty()) return std::unexpected(operation_error{"recording has no frames"});
    const auto begin = input.frames.front().start_stamp_ns;
    auto minimum_stamp = begin;
    std::uint64_t end{};
    for (const auto& frame : input.frames) {
        minimum_stamp = std::min(minimum_stamp, frame.start_stamp_ns);
        end = std::max(end, frame.end_stamp_ns);
    }
    for (const auto* trajectory : {&input.ground_truth, &input.odometry})
    {
        if (trajectory->empty() && trajectory == &input.ground_truth && input.type == dataset_wire::recorded_type) continue;
        if (trajectory->empty() || trajectory->front().stamp_ns > minimum_stamp || trajectory->back().stamp_ns < end)
            return std::unexpected(operation_error{"trajectory does not cover every scan"});
    }
    return {};
}
}

operation_result<pose> sample_pose(
    std::span<const pose_sample> trajectory, std::uint64_t stamp_ns)
{
    if (trajectory.empty())
        return std::unexpected(operation_error{"cannot sample an empty trajectory"});
    if (stamp_ns < trajectory.front().stamp_ns || stamp_ns > trajectory.back().stamp_ns)
        return std::unexpected(operation_error{"pose timestamp is outside trajectory coverage"});
    const auto upper = std::lower_bound(trajectory.begin(), trajectory.end(), stamp_ns,
        [](const auto& sample, std::uint64_t stamp) { return sample.stamp_ns < stamp; });
    if (upper == trajectory.end())
        return std::unexpected(operation_error{"invalid trajectory ordering"});
    const auto upper_valid = validate_pose_sample(*upper);
    if (!upper_valid) return std::unexpected(upper_valid.error());
     if (upper == trajectory.begin() || upper->stamp_ns == stamp_ns) return upper->pose;
    const auto& left = *(upper - 1);
    const auto left_valid = validate_pose_sample(left);
    if (!left_valid) return std::unexpected(left_valid.error());
    if (left.stamp_ns >= upper->stamp_ns)
        return std::unexpected(operation_error{"invalid trajectory interval"});
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

operation_result<std::span<const pose_sample>> pose_stream(
    const dataset& input, pose_source source)
{
    const auto* result = source == pose_source::ground_truth ? &input.ground_truth : &input.odometry;
    if ((source != pose_source::ground_truth && source != pose_source::odometry) || result->empty())
        return std::unexpected(operation_error{"requested trajectory is unavailable"});
     return *result;
}
}

#include <charconv>

namespace covoxslam
{
namespace
{
operation_result<std::size_t> frame_beam_count(const frame_descriptor& frame);

operation_result<std::uint64_t> input_structure_hash(
    const dataset& input, const detail::json_value& metadata)
{
    std::vector<std::filesystem::path> paths{input.manifest_path};
    for (const auto key : {"sensor_file", "world_file", "timeline_file"})
    {
        if (!metadata.members.contains(key)) continue;
        const auto member = metadata.member(key);
        if (!member) return std::unexpected(member.error());
        const auto relative = member->get().string();
        if (!relative) return std::unexpected(relative.error());
        const auto path = detail::dataset_path(input.root, relative->get());
        if (!path) return std::unexpected(path.error());
        paths.push_back(*path);
    }
    for (const auto& frame : input.frames)
    {
        const auto source = hash_file(frame.metadata_path);
        if (!source) return std::unexpected(source.error());
        if (source->content_hash != frame.metadata_content_hash)
            return std::unexpected(operation_error{"frame metadata changed after parsing"});
        paths.push_back(frame.metadata_path);
    }
    auto hash = hash_basis;
    const auto append = [&hash](const auto& value) {
        hash = hash_bytes({reinterpret_cast<const std::uint8_t*>(&value), sizeof(value)}, hash);
    };
    for (const auto& path : paths) {
        const auto relative = path.lexically_relative(input.root).generic_string();
        const auto source = hash_file(path);
        if (!source) return std::unexpected(source.error());
        append(static_cast<std::uint64_t>(relative.size()));
        append(hash_bytes({reinterpret_cast<const std::uint8_t*>(relative.data()), relative.size()}));
        append(source->identity.size);
        append(source->content_hash);
    }
    for (const auto key : {"trajectory_ground_truth_file", "odometry_file"})
    {
        if (!metadata.members.contains(key)) continue;
        const auto member = metadata.member(key);
        if (!member) return std::unexpected(member.error());
        const auto relative = member->get().string();
        if (!relative) return std::unexpected(relative.error());
        const auto path = detail::dataset_path(input.root, relative->get());
        if (!path) return std::unexpected(path.error());
        const auto identity = file_identity(*path);
        if (!identity) return std::unexpected(identity.error());
        append(identity->size);
        append(identity->modified);
    }
    return hash;
}

operation_status prepare_payload_index(dataset& input)
{
    decltype(input.payload_index) payload_index;
    for (const auto& frame : input.frames)
    {
        const auto count = frame_beam_count(frame);
        if (!count) return std::unexpected(count.error());
        const std::array<std::pair<std::filesystem::path, std::uint64_t>, 5> payloads{{
            {frame.raw_xyz_path, *count * dataset_wire::xyz_width},
            {frame.deskewed_xyz_path, *count * dataset_wire::xyz_width},
             {frame.range_path, *count * dataset_wire::range_width},
            {frame.status_path, *count * dataset_wire::status_width},
            {frame.time_offset_ns_path, frame.columns * dataset_wire::time_offset_width}}};
        for (const auto& [path, size] : payloads) {
            if (path.empty()) continue;
            const auto resolved = detail::dataset_path(input.root, path.lexically_relative(input.root));
            if (!resolved) return std::unexpected(resolved.error());
            const auto identity = file_identity(*resolved);
            if (!identity) return std::unexpected(identity.error());
            if (identity->size != size)
                return std::unexpected(operation_error{"indexed payload size disagrees with frame metadata"});
            payload_index.emplace(*resolved, payload_cache_descriptor{*identity, 0});
        }
    }
    input.payload_index = std::move(payload_index);
    return {};
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
    std::array<char, dataset_wire::fingerprint_digit_capacity> hexadecimal;
    const auto result = std::to_chars(hexadecimal.data(), hexadecimal.data() + hexadecimal.size(),
                                     hash, dataset_wire::fingerprint_base);
    input.content_fingerprint = dataset_wire::fingerprint_prefix + std::string(hexadecimal.data(), result.ptr);
}

operation_status index_dataset_payloads(dataset& input, std::uint64_t structure_hash)
{
    auto previous_index = std::move(input.payload_index);
    const auto prepared = prepare_payload_index(input);
    if (!prepared) {
        input.payload_index = std::move(previous_index);
        return std::unexpected(prepared.error());
    }
    auto payload_index = std::move(input.payload_index);
    input.payload_index = std::move(previous_index);
    for (auto& [path, payload] : payload_index)
    {
        const auto verified = hash_file(path);
        if (!verified) return std::unexpected(verified.error());
        if (verified->identity != payload.identity)
            return std::unexpected(operation_error{"payload changed while the input index was built"});
        payload = *verified;
    }
    input.payload_index = std::move(payload_index);
    update_content_fingerprint(input, structure_hash);
    return {};
}

operation_status verify_frame_bytes(const dataset& input, const std::filesystem::path& path,
                                    std::span<const std::uint8_t> bytes)
{
    if (input.payload_index.empty()) return {};
    const auto entry = input.payload_index.find(path);
    if (entry == input.payload_index.end() || entry->second.identity.size != bytes.size() ||
        entry->second.content_hash != hash_bytes(bytes))
        return std::unexpected(operation_error{"frame payload does not match the verified input index"});
    return {};
}

operation_status append_cached_trajectory(cache_buffer& buffer, std::span<const pose_sample> trajectory)
{
    if (trajectory.size() > maximum_trajectory_samples)
        return std::unexpected(operation_error{"trajectory cache exceeds sample limit"});
    auto appended = buffer.append(static_cast<std::uint64_t>(trajectory.size()));
    if (!appended) return std::unexpected(appended.error());
    for (const auto& sample : trajectory) {
        const auto valid = validate_pose_sample(sample);
        if (!valid) return std::unexpected(valid.error());
        appended = buffer.append(sample.stamp_ns);
        if (!appended) return std::unexpected(appended.error());
        const auto& p = sample.pose.translation;
        const auto& q = sample.pose.rotation;
        for (const float component : {p.x, p.y, p.z, q.x, q.y, q.z, q.w}) {
            appended = buffer.append(component);
            if (!appended) return std::unexpected(appended.error());
        }
    }
    return {};
}

operation_status write_input_cache(const dataset& input, std::uint64_t structure_hash)
{
    cache_buffer buffer;
    auto appended = buffer.append(dataset_wire::cache_magic);
    if (!appended) return std::unexpected(appended.error());
    appended = buffer.append(structure_hash);
    if (!appended) return std::unexpected(appended.error());
    appended = buffer.append(static_cast<std::uint64_t>(input.payload_index.size()));
    if (!appended) return std::unexpected(appended.error());
    for (const auto& [path, payload] : input.payload_index) {
        const auto identity = file_identity(path);
        if (!identity) return std::unexpected(identity.error());
        if (*identity != payload.identity)
            return std::unexpected(operation_error{"payload changed while the input cache was built"});
        appended = buffer.append(payload.identity.size);
        if (!appended) return std::unexpected(appended.error());
        appended = buffer.append(payload.identity.modified);
        if (!appended) return std::unexpected(appended.error());
        appended = buffer.append(payload.content_hash);
        if (!appended) return std::unexpected(appended.error());
    }
    auto trajectory = append_cached_trajectory(buffer, input.ground_truth);
    if (!trajectory) return std::unexpected(trajectory.error());
    trajectory = append_cached_trajectory(buffer, input.odometry);
    if (!trajectory) return std::unexpected(trajectory.error());
    return write_cache_bytes(input.root, buffer);
}
}
}

namespace covoxslam
{
namespace
{
operation_result<std::vector<pose_sample>> read_cached_trajectory(cache_buffer& buffer)
{
    const auto count = buffer.take<std::uint64_t>();
    if (!count) return std::unexpected(count.error());
    if (*count > maximum_trajectory_samples ||
        *count > buffer.remaining() / dataset_wire::cached_trajectory_record_width)
        return std::unexpected(operation_error{"invalid trajectory cache sample count"});
    std::vector<pose_sample> result;
     result.reserve(static_cast<std::size_t>(*count));
    for (std::uint64_t index = 0; index < *count; ++index)
    {
        pose_sample sample;
        const auto stamp = buffer.take<std::uint64_t>();
        if (!stamp) return std::unexpected(stamp.error());
        sample.stamp_ns = *stamp;
        auto& p = sample.pose.translation;
        auto& q = sample.pose.rotation;
        for (float* component : {&p.x, &p.y, &p.z, &q.x, &q.y, &q.z, &q.w}) {
            const auto value = buffer.take<float>();
            if (!value) return std::unexpected(value.error());
            *component = *value;
        }
        const auto valid = validate_pose_sample(sample);
        if (!valid) return std::unexpected(valid.error());
        if (!result.empty() && sample.stamp_ns <= result.back().stamp_ns)
            return std::unexpected(operation_error{"cached trajectory timestamps are not strictly increasing"});
         result.push_back(sample);
    }
    return result;
}

operation_result<bool> read_input_cache(
    dataset& input, std::uint64_t structure_hash, bool ground_truth_expected)
{
    const auto path = detail::dataset_path(input.root, dataset_wire::cache_destination_filename);
    if (!path) return std::unexpected(path.error());
    auto buffer = read_cache_bytes(*path);
    if (!buffer) return std::unexpected(buffer.error());
    const auto saved_structure = buffer->take<std::uint64_t>();
    if (!saved_structure) return std::unexpected(saved_structure.error());
    const auto payload_count = buffer->take<std::uint64_t>();
    if (!payload_count) return std::unexpected(payload_count.error());
    if (*saved_structure != structure_hash || *payload_count != input.payload_index.size()) return false;
    auto payloads = input.payload_index;
    for (auto& [path, payload] : payloads)
    {
        const auto size = buffer->take<std::uint64_t>();
        if (!size) return std::unexpected(size.error());
        const auto modified = buffer->take<std::int64_t>();
        if (!modified) return std::unexpected(modified.error());
        const cache_file_identity saved{*size, *modified};
        if (saved != payload.identity) return false;
        const auto hash = buffer->take<std::uint64_t>();
        if (!hash) return std::unexpected(hash.error());
        payload.content_hash = *hash;
    }
    auto ground_truth = read_cached_trajectory(*buffer);
    if (!ground_truth) return std::unexpected(ground_truth.error());
    auto odometry = read_cached_trajectory(*buffer);
    if (!odometry) return std::unexpected(odometry.error());
    if (buffer->remaining() != 0 || odometry->empty() ||
        ground_truth_expected == ground_truth->empty()) return false;
    for (const auto& [path, payload] : payloads) {
        const auto identity = file_identity(path);
        if (!identity) return std::unexpected(identity.error());
        if (*identity != payload.identity) return false;
    }
    input.payload_index = std::move(payloads);
    input.ground_truth = std::move(*ground_truth);
    input.odometry = std::move(*odometry);
    return true;
}

operation_status load_dataset_inputs(dataset& input, const detail::json_value& metadata,
                                     dataset_load_options options)
{
    if (options.verification != dataset_verification_mode::prefer_cache &&
        options.verification != dataset_verification_mode::force_full)
        return std::unexpected(operation_error{"unsupported input verification mode"});
    const auto structure_hash = input_structure_hash(input, metadata);
    if (!structure_hash) return std::unexpected(structure_hash.error());
    const auto prepared = prepare_payload_index(input);
    if (!prepared) return std::unexpected(prepared.error());
    bool cached{};
    if (options.verification == dataset_verification_mode::prefer_cache) {
        const auto loaded = read_input_cache(
            input, *structure_hash, metadata.members.contains("trajectory_ground_truth_file"));
        cached = loaded && *loaded;
    }
    if (!cached) {
        auto loaded = load_json_trajectories(input, metadata);
        if (!loaded) return std::unexpected(loaded.error());
        const auto indexed = index_dataset_payloads(input, *structure_hash);
        if (!indexed) return std::unexpected(indexed.error());
    }
    const auto final_structure_hash = input_structure_hash(input, metadata);
    if (!final_structure_hash) return std::unexpected(final_structure_hash.error());
    if (*final_structure_hash != *structure_hash)
        return std::unexpected(operation_error{"recording metadata changed while inputs were loaded"});
    const auto coverage = validate_trajectory_coverage(input);
    if (!coverage) return std::unexpected(coverage.error());
    input.input_source = cached ? dataset_input_source::validated_cache : dataset_input_source::full_scan;
    update_content_fingerprint(input, *structure_hash);
    return {};
}
}
}
