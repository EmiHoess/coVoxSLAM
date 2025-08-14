#pragma once
#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/input_timing.hpp>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>
namespace covoxslam {
struct dataset; struct frame_descriptor {
    std::uint64_t metadata_content_hash{};
    int frame_id{}, submap_id{-1};
    std::uint64_t start_stamp_ns{}, end_stamp_ns{};
    std::size_t rows{}, columns{}, beam_count{}, hit_count{}, dropout_count{}, invalid_count{}, outlier_count{};
    pose ground_truth_start{}, ground_truth_end{}, odometry_start{}, odometry_end{};
    std::filesystem::path metadata_path, raw_xyz_path, deskewed_xyz_path, range_path, time_offset_ns_path, status_path;
};
enum class beam_status : std::uint8_t { invalid_no_return = 0, valid_hit = 1, synthetic_dropout = 2, valid_outlier = 3 };
enum class point_source { raw, deskewed };
struct frame_load_options { bool raw_xyz{true}, deskewed_xyz{false}, range{true}, status{true}, time_offset_ns{true};
    std::size_t max_payload_bytes{256U * 1024U * 1024U}; };
struct organized_frame {
    frame_descriptor descriptor;
    input_timing timing;
    std::vector<vec3f> raw_xyz, deskewed_xyz;
    std::vector<std::uint32_t> range_mm, time_offset_ns;
    std::vector<std::uint8_t> status;
    std::size_t flat_index(std::size_t ring, std::size_t column) const;
    const vec3f& point(point_source source, std::size_t ring, std::size_t column) const;
    std::uint64_t beam_stamp_ns(std::size_t column) const;
    bool valid_return(std::size_t ring, std::size_t column) const;
};
operation_result<frame_descriptor> read_frame_descriptor(const std::filesystem::path& root,
    const std::filesystem::path& metadata_path);
operation_result<organized_frame> load_frame(
    const dataset& input, std::size_t frame_index, frame_load_options options = {}); }
