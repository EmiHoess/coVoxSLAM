#pragma once
#include <covoxslam/input_timing.hpp>

#include <covoxslam/types.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace covoxslam {

struct pose_sample {
    std::uint64_t stamp_ns{};
     covoxslam::pose pose{};
};

struct sensor {
    std::string model;
    std::string acquisition_model;
    std::size_t rings{};
    std::size_t columns{};
    double scan_rate_hz{};
    std::uint64_t scan_period_ns{};
     double azimuth_start_rad{};
    double azimuth_coverage_rad{};
    bool counter_clockwise{true};
    std::vector<float> vertical_angles_rad;
    float min_range_m{};
    float max_range_m{};
    pose body_from_lidar{};
};

enum class world_primitive_kind {
     axis_aligned_plane,
    axis_aligned_box,
    sphere
};

enum class axis {
    x,
    y,
    z
};

 struct world_primitive {
    std::string name;
    world_primitive_kind kind{world_primitive_kind::axis_aligned_plane};
    covoxslam::axis axis{covoxslam::axis::z};
    float coordinate{};
    vec3f min{};
    vec3f max{};
    vec3f center{};
    float radius{};
};

struct world_description {
     std::string name;
    std::string type;
    std::string primitive_set;
    std::string frame{"world"};
    vec3f bounds_min{};
    vec3f bounds_max{};
    std::vector<world_primitive> primitives;
};

struct frame_descriptor {
    std::uint64_t metadata_content_hash{};
    int frame_id{};
     int submap_id{-1};
    std::uint64_t start_stamp_ns{};
    std::uint64_t end_stamp_ns{};
    std::size_t rows{};
    std::size_t columns{};
    std::size_t beam_count{};
    std::size_t hit_count{};
    std::size_t dropout_count{};
    std::size_t invalid_count{};
    std::size_t  outlier_count{};
    pose ground_truth_start{};
    pose ground_truth_end{};
    pose odometry_start{};
    pose odometry_end{};
    std::filesystem::path metadata_path;
    std::filesystem::path raw_xyz_path;
    std::filesystem::path deskewed_xyz_path;
    std::filesystem::path range_path;
     std::filesystem::path time_offset_ns_path;
    std::filesystem::path status_path;
};

enum class pose_source { ground_truth, odometry };

enum class dataset_verification_mode
{
    prefer_cache,
     force_full
};

enum class dataset_input_source { full_scan, validated_cache };

struct dataset_load_options
{
    dataset_verification_mode verification{dataset_verification_mode::prefer_cache};
    bool update_cache{true};
};

struct cache_file_identity {
    std::uint64_t size{};
    std::int64_t modified{};
     bool operator==(const cache_file_identity&) const noexcept = default;
}
;

struct payload_cache_descriptor
{
    cache_file_identity identity;
    std::uint64_t content_hash{};
};

struct dataset {
    std::filesystem::path root;
    std::string schema_version;
    std::string name;
    std::string type;
    covoxslam::sensor sensor;
    world_description world;
     std::vector<frame_descriptor> frames;
    std::vector<pose_sample> ground_truth;
    std::vector<pose_sample> odometry;
    std::map<std::filesystem::path, payload_cache_descriptor> payload_index;
    dataset_input_source input_source{dataset_input_source::full_scan};
    std::string content_fingerprint;
};

enum class beam_status : std::uint8_t
{
    invalid_no_return = 0,
    valid_hit = 1,
     synthetic_dropout = 2,
    valid_outlier = 3
};

enum class point_source {
    raw,
    deskewed
}
;

struct frame_load_options
{
    bool raw_xyz{true};
    bool deskewed_xyz{false};
     bool range{true};
    bool status{true};
    bool time_offset_ns{true};
    std::size_t max_payload_bytes{256U * 1024U * 1024U};
};

struct organized_frame
{
    frame_descriptor descriptor;
    input_timing timing;
    std::vector<vec3f> raw_xyz;
    std::vector<vec3f> deskewed_xyz;
     std::vector<std::uint32_t> range_mm;
    std::vector<std::uint32_t> time_offset_ns;
    std::vector<std::uint8_t> status;
    std::size_t flat_index(std::size_t ring, std::size_t column) const;
    const vec3f& point(point_source source, std::size_t ring, std::size_t column) const;
    std::uint64_t beam_stamp_ns(std::size_t column) const;
    bool valid_return(std::size_t ring, std::size_t column) const;
};

frame_descriptor read_frame_descriptor(
    const std::filesystem::path& root, const std::filesystem::path& metadata_path);
organized_frame load_frame(
    const dataset& input, std::size_t frame_index, frame_load_options options = {});

dataset load_dataset(const std::filesystem::path& root, dataset_load_options options = {});
pose sample_pose(std::span<const pose_sample> trajectory, std::uint64_t stamp_ns);
std::span<const pose_sample> pose_stream(const dataset& input, pose_source source);

}
