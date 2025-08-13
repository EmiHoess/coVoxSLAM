#pragma once
#include <covoxslam/input/calibration.hpp>
#include <covoxslam/input/lidar.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>
namespace covoxslam {
struct pose_sample { std::uint64_t stamp_ns{}; covoxslam::pose pose{}; };
enum class world_primitive_kind { axis_aligned_plane, axis_aligned_box, sphere };
enum class axis { x, y, z };
struct world_primitive { std::string name; world_primitive_kind kind{world_primitive_kind::axis_aligned_plane};
    covoxslam::axis axis{covoxslam::axis::z}; float coordinate{}; vec3f min{}, max{}, center{}; float radius{}; };
struct world_description { std::string name, type, primitive_set, frame{"world"};
    vec3f bounds_min{}, bounds_max{}; std::vector<world_primitive> primitives; };
enum class pose_source { ground_truth, odometry };
enum class dataset_verification_mode { prefer_cache, force_full };
enum class dataset_input_source { full_scan, validated_cache };
struct dataset_load_options { dataset_verification_mode verification{dataset_verification_mode::prefer_cache}; bool update_cache{true}; };
struct cache_file_identity { std::uint64_t size{}; std::int64_t modified{}; bool operator==(const cache_file_identity&) const noexcept = default; };
struct payload_cache_descriptor { cache_file_identity identity; std::uint64_t content_hash{}; };
struct dataset {
    std::filesystem::path root, manifest_path;
    std::string schema_version, name, type;
    covoxslam::sensor sensor;
    world_description world;
    std::vector<frame_descriptor> frames;
    std::vector<pose_sample> ground_truth, odometry;
    std::map<std::filesystem::path, payload_cache_descriptor> payload_index;
    dataset_input_source input_source{dataset_input_source::full_scan};
    std::string content_fingerprint;
};
operation_result<dataset> load_dataset(const std::filesystem::path& root, dataset_load_options options = {});
pose sample_pose(std::span<const pose_sample> trajectory, std::uint64_t stamp_ns);
std::span<const pose_sample> pose_stream(const dataset& input, pose_source source); }
