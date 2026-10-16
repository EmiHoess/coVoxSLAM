#pragma once

#include <cstddef>
#include <cstdint>

namespace covoxslam::detail::dataset_format
{
inline constexpr char schema[] = "covox.spinning_lidar.v2";
inline constexpr char frame_document[] = "spinning_lidar_frame";
inline constexpr char dataset_document[] = "spinning_lidar_dataset";
inline constexpr char sensor_document[] = "spinning_lidar_sensor";
inline constexpr char trajectory_record_document[] = "spinning_lidar_trajectory_record";
inline constexpr char recorded_type[] = "recorded_spinning_lidar";
inline constexpr char synthetic_type[] = "synthetic_spinning_lidar";
inline constexpr char json_encoding[] = "json";
inline constexpr char toml_encoding[] = "toml";
inline constexpr char toml_extension[] = ".toml";
inline constexpr char json_manifest[] = "dataset.json";
inline constexpr char toml_manifest[] = "dataset.toml";
inline constexpr std::uint64_t cache_magic = 0x31584449434f5658ULL;
inline constexpr std::size_t cache_minimum_prefix_bytes = 16;
inline constexpr char cache_staging_directory[] = ".covox_input_cache_write";
inline constexpr char cache_staging_filename[] = "input.bin";
inline constexpr char cache_destination_filename[] = ".covox_input_cache_v1.bin";
inline constexpr char fingerprint_prefix[] = "dataset-native-v1-";
inline constexpr int fingerprint_base = 16;
inline constexpr std::size_t fingerprint_digit_capacity = 16;
inline constexpr std::size_t float32_width = 4;
inline constexpr std::size_t xyz_width = 12;
inline constexpr std::size_t uint32_width = 4;
inline constexpr std::size_t range_width = uint32_width;
inline constexpr std::size_t time_offset_width = uint32_width;
inline constexpr std::size_t status_width = 1;
inline constexpr std::size_t cached_trajectory_record_width = 36;
inline constexpr std::size_t hash_chunk_bytes = 65536;

static_assert(sizeof(float) == float32_width);
static_assert(sizeof(std::uint32_t) == uint32_width);
static_assert(xyz_width == 3 * float32_width);
static_assert(cached_trajectory_record_width == sizeof(std::uint64_t) + 7 * float32_width);
}
