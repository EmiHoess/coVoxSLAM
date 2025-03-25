#pragma once
#include <covoxslam/storage/selection.hpp>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>

namespace covoxslam::metrics
{
using namespace storage;
using clock = std::chrono::steady_clock;
enum class copy_method { scalar, range, packed, unpacked };
struct measurement_options
{
    int warmup{3}, repeats{7};
    copy_method method{copy_method::scalar};
    std::size_t batch_records{32};
    bool fixed_batch{};
};
struct measurement
{
    std::vector<double> samples_ms;
    double setup_ms{}, upload_ms{}, download_ms{};
    std::size_t mismatch_count{}, bytes{}, copy_calls{};
    std::uint64_t output_hash{};
};
inline double elapsed(clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}
template<class t> std::uint64_t hash_values(std::span<const t> values)
{
    std::uint64_t hash = 14695981039346656037ULL;

    for (auto byte : std::as_bytes(values)) { hash ^= std::to_integer<unsigned char>(byte); hash *= 1099511628211ULL; }
    return hash;
}
template<schema s> operation_result<std::vector<aos_tile<s>>> load_tiles(
    const std::filesystem::path& root, std::size_t last_submap, std::size_t maximum_tiles = 100000)
{
    static_assert(s == schema::tsdf || s == schema::esdf);
    static_assert(std::endian::native == std::endian::little);
    static_assert(sizeof(aos_tile<s>) == sizeof(value<s>) * voxels_per_tile);
    if (last_submap > 100000) return std::unexpected(operation_error{"submap limit exceeded"});
    std::vector<aos_tile<s>> result;
    float common_voxel{}, common_band{};

    for (std::size_t id = 0; id <= last_submap; ++id)
    {
        const auto suffix = s == schema::tsdf ? ".tsdf.bin" : ".esdf.bin";
        std::ifstream input(root / "submaps" / ("submap_" + std::to_string(id) + suffix), std::ios::binary | std::ios::ate);
        if (!input) return std::unexpected(operation_error{"cannot open frozen map"});
        const auto bytes = input.tellg();
        input.seekg(0);
        char magic[8]{};
        std::uint32_t version{}, edge{};
        float voxel{}, band{};
        std::uint64_t count{};
        const auto read = [&](auto& v) { input.read(reinterpret_cast<char*>(&v), sizeof(v)); };
        read(magic); read(version); read(edge); read(voxel); read(band); read(count);
        const auto expected = s == schema::tsdf ? "CVXTSDF1" : "CVXESDF1";

        if (!input || std::memcmp(magic, expected, 8) || version != 1 || edge != 8 ||
            !std::isfinite(voxel) || !std::isfinite(band) || voxel <= 0 || band <= 0 ||
            (id != 0 && (common_voxel != voxel || common_band != band)) ||
            count > maximum_tiles - result.size() || bytes != std::streamoff(32 + count * (12 + sizeof(aos_tile<s>))))
            return std::unexpected(operation_error{"invalid frozen map header or size"});
        common_voxel = voxel; common_band = band;
        const auto first = result.size();
        result.resize(first + std::size_t(count));
        std::array<std::int32_t, 3> previous{};

        for (std::size_t tile = 0; tile < count; ++tile) {
            std::array<std::int32_t, 3> key{};
            read(key);
            auto& record = result[first + tile];
            read(record);
            if (!input || (tile && std::tie(key[2], key[1], key[0]) <= std::tie(previous[2], previous[1], previous[0])))
                return std::unexpected(operation_error{"invalid frozen tile record"});
            previous = key;

            for (const auto& v : record.voxels)
            {
                if constexpr (s == schema::tsdf) {
                    if (!std::isfinite(v.distance) || !std::isfinite(v.weight) || v.weight < 0 || (v.flags & ~3U))
                        return std::unexpected(operation_error{"invalid frozen TSDF value"});
                }
                else if (!std::isfinite(v.signed_distance_m) || (v.flags & ~7U))
                    return std::unexpected(operation_error{"invalid frozen ESDF value"});
            }
        }
        if (input.peek() != std::char_traits<char>::eof() || input.bad())
            return std::unexpected(operation_error{"frozen map changed while reading"});
    }
    if (result.empty()) return std::unexpected(operation_error{"frozen map has no tiles"});
    return result;
}
template<class t> auto record_value(const t& record)
{
    if constexpr (requires { gather_record(record); }) return gather_record(record);
    else return record;
}
template<schema s, class view_type> void field_query(view_type view, std::span<float> output)
{
    for (std::size_t tile = 0; tile < view.size(); ++tile)
        for (std::size_t voxel = 0; voxel < voxels_per_tile; ++voxel) {
            const auto v = view[tile].voxels[voxel];
            if constexpr (s == schema::tsdf) output[tile * 512 + voxel] = (v.flags & 1U) ? v.distance * v.weight : 0;
            else output[tile * 512 + voxel] = (v.flags & 1U) ? v.signed_distance_m : 0;
        }
}
template<schema s, class view_type> void record_copy(view_type view, std::span<value<s>> output)
{
    for (std::size_t tile = 0; tile < view.size(); ++tile)
        for (std::size_t voxel = 0; voxel < voxels_per_tile; ++voxel)
            output[tile * 512 + voxel] = record_value(view[tile].voxels[voxel]);
}
}
