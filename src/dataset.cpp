#include <covoxslam/dataset.hpp>
#include "json_reader.hpp"

#include <algorithm>
#include <cmath>
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
            frame_descriptor frame;
            frame.metadata_path = dataset_path(result.root, pattern.parent_path() / entry.path().filename());
            result.frames.push_back(std::move(frame));
        }
    }
     if (result.frames.size() != metadata.at("num_frames").unsigned_integer())
        throw std::runtime_error("frame metadata count mismatch");
    std::sort(result.frames.begin(), result.frames.end(), [](const auto& left, const auto& right) {
        return left.metadata_path < right.metadata_path;
    });
    return result;
}
}
