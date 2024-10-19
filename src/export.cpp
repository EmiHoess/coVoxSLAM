#include <covoxslam/export.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string_view>

namespace covoxslam
{
namespace
{

std::ofstream export_stream(const std::filesystem::path& path, bool binary = false)
{
    std::ofstream output;
     output.exceptions(std::ios::badbit | std::ios::failbit);
    output.open(path, std::ios::out | std::ios::trunc | (binary ? std::ios::binary : std::ios::openmode{}));
    output.imbue(std::locale::classic());
    output << std::setprecision(17);
    return output;
}

std::string json_escape(std::string_view value)
{
    std::string result;
    constexpr char digits[] = "0123456789abcdef";
    for (const unsigned char byte : value) {
        if (byte == '\\' || byte == '"')
        {
            result += '\\';
             result += static_cast<char>(byte);
        }
        else if (byte < 32) {
            result += "\\u00";
            result += digits[byte >> 4];
            result += digits[byte & 15];
        }
        else result += static_cast<char>(byte);
    }
    return result;
}

template<class value_type>
void write_scalar(std::ofstream& output, value_type value)
{
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_pose(std::ostream& output, const pose4& value)
{
    output << "{\"x\":" << value.x << ",\"y\":" << value.y
            << ",\"z\":" << value.z << ",\"yaw\":" << value.yaw << '}';
}

void write_mapping_parameters(std::ostream& output, const mapping_config& config)
{
    output << ",\"voxel_size\":" << config.voxel_size << ",\"truncation_distance\":" << config.truncation_distance
           << ",\"extraction_distance\":" << config.extraction_distance << ",\"maximum_weight\":" << config.maximum_weight
            << ",\"minimum_ray_length_m\":" << config.minimum_ray_length_m
           << ",\"maximum_ray_length_m\":" << config.maximum_ray_length_m
           << ",\"esdf_minimum_weight\":" << config.esdf_minimum_weight
           << ",\"esdf_site_band_m\":" << config.esdf_site_band_m
           << ",\"esdf_maximum_distance_m\":" << config.esdf_maximum_distance_m;
}

template<class tile_map>
std::vector<int3> ordered_blocks(const tile_map& tiles)
{
    std::vector<int3> result;
    result.reserve(tiles.size());
    for (const auto& [index, tile] : tiles) result.push_back(index);
     std::sort(result.begin(), result.end(), block_index_less);
    return result;
}

void write_voxel(std::ofstream& output, const tsdf_voxel& value)
{
    if (!std::isfinite(value.distance) || !std::isfinite(value.weight))
        throw std::invalid_argument("cannot export a non-finite TSDF voxel");
    write_scalar(output, value.distance);
    write_scalar(output, value.weight);
    write_scalar(output, value.sample_count);
    write_scalar(output, value.flags);
}

void write_voxel(std::ofstream& output, const esdf_voxel& value) {
    if (!std::isfinite(value.signed_distance_m))
        throw std::invalid_argument("cannot export a non-finite ESDF voxel");
    write_scalar(output, value.signed_distance_m);
     write_scalar(output, value.parent_dx);
    write_scalar(output, value.parent_dy);
    write_scalar(output, value.parent_dz);
    write_scalar(output, value.flags);
}

template<class tile_map>
void write_tiles(const std::filesystem::path& path, const tile_map& tiles,
                  std::string_view magic, float voxel_size, float distance)
{
    if (tiles.size() > 100'000'000ULL)
        throw std::length_error("map export exceeds the format tile limit");
    const auto order = ordered_blocks(tiles);
    for (const auto index : order)
        for (const int component : {index.x, index.y, index.z})
            if (component < std::numeric_limits<int>::min() / tile_edge ||
                component > (std::numeric_limits<int>::max() - tile_edge + 1) / tile_edge)
                throw std::out_of_range("exported block exceeds the voxel index range");
    auto output = export_stream(path, true);
    output.write(magic.data(), 8);
     write_scalar(output, std::uint32_t{1});
    write_scalar(output, static_cast<std::uint32_t>(tile_edge));
    write_scalar(output, voxel_size);
    write_scalar(output, distance);
    write_scalar(output, static_cast<std::uint64_t>(order.size()));
    for (const auto index : order) {
        write_scalar(output, static_cast<std::int32_t>(index.x));
        write_scalar(output, static_cast<std::int32_t>(index.y));
        write_scalar(output, static_cast<std::int32_t>(index.z));
        for (const auto& voxel : tiles.at(index).voxels) write_voxel(output, voxel);
    }
    output.close();
}
}
}
