#include <covoxslam/io/export.hpp>
#include <covoxslam/voxel/flags.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <string_view>

namespace covoxslam
{
namespace
{

struct tile_format_descriptor
{
    std::array<char, 8> magic;
};

inline constexpr tile_format_descriptor tsdf_tile_format{{'C', 'V', 'X', 'T', 'S', 'D', 'F', '1'}};
inline constexpr tile_format_descriptor esdf_tile_format{{'C', 'V', 'X', 'E', 'S', 'D', 'F', '1'}};
inline constexpr std::uint32_t tile_binary_version = 1;
inline constexpr std::uint64_t maximum_export_tile_count = 100'000'000ULL;
inline constexpr std::uint64_t maximum_exact_json_integer = 9'007'199'254'740'991ULL;
inline constexpr char run_schema[] = "covoxslam.run.v1";
inline constexpr char poses_filename[] = "poses.jsonl";
inline constexpr char submaps_filename[] = "submaps.jsonl";
inline constexpr char submaps_directory_name[] = "submaps";
inline constexpr char submap_filename_prefix[] = "submap_";
inline constexpr char tsdf_filename_suffix[] = ".tsdf.bin";
inline constexpr char esdf_filename_suffix[] = ".esdf.bin";
inline constexpr char run_filename[] = "run.json";
inline constexpr char run_pending_filename[] = "run.json.pending";

operation_result<std::ofstream> export_stream(const std::filesystem::path& path,
                                               std::string_view diagnostic, bool binary = false)
{
    std::ofstream output;
    output.open(path, std::ios::out | std::ios::trunc | (binary ? std::ios::binary : std::ios::openmode{}));
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (!output) return std::unexpected(operation_error{std::string(diagnostic)});
    return output;
}

class export_directory_cleanup
{
    std::filesystem::path directory;
    bool armed{true};
public:
    explicit export_directory_cleanup(std::filesystem::path path) : directory(std::move(path)) {}
    ~export_directory_cleanup()
    {
        if (!armed) return;
        std::error_code error;
        std::filesystem::remove(directory / poses_filename, error);
        error.clear();
        std::filesystem::remove(directory / submaps_filename, error);
        error.clear();
        std::filesystem::remove(directory / submaps_directory_name, error);
        error.clear();
        std::filesystem::remove(directory, error);
    }
    void release() { armed = false; }
};

class export_file_cleanup
{
    std::filesystem::path path;
    bool armed{true};
public:
    explicit export_file_cleanup(std::filesystem::path value) : path(std::move(value)) {}
    ~export_file_cleanup()
    {
        if (!armed) return;
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    void release() { armed = false; }
};

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

operation_status write_voxel(std::ofstream& output, const tsdf_voxel& value)
{
    if (!std::isfinite(value.distance) || !std::isfinite(value.weight))
        return std::unexpected(operation_error{"cannot export a non-finite TSDF voxel"});
    write_scalar(output, value.distance);
    write_scalar(output, value.weight);
    write_scalar(output, value.sample_count);
    write_scalar(output, value.flags);
    if (!output) return std::unexpected(operation_error{"cannot write TSDF voxel export"});
    return {};
}

operation_status write_voxel(std::ofstream& output, const esdf_voxel& value) {
    if (!std::isfinite(value.signed_distance_m))
        return std::unexpected(operation_error{"cannot export a non-finite ESDF voxel"});
    write_scalar(output, value.signed_distance_m);
     write_scalar(output, value.parent_dx);
    write_scalar(output, value.parent_dy);
    write_scalar(output, value.parent_dz);
    write_scalar(output, value.flags);
    if (!output) return std::unexpected(operation_error{"cannot write ESDF voxel export"});
    return {};
}

template<class tile_map>
operation_status write_tiles(const std::filesystem::path& path, const tile_map& tiles,
                             tile_format_descriptor format, float voxel_size, float distance)
{
    if (tiles.size() > maximum_export_tile_count)
        return std::unexpected(operation_error{"map export exceeds the format tile limit"});
    const auto order = ordered_blocks(tiles);
    for (const auto index : order)
        for (const int component : {index.x, index.y, index.z})
            if (component < std::numeric_limits<int>::min() / tile_edge ||
                component > (std::numeric_limits<int>::max() - tile_edge + 1) / tile_edge)
                return std::unexpected(operation_error{"exported block exceeds the voxel index range"});
    auto opened = export_stream(path, "cannot open map tile export", true);
    if (!opened) return std::unexpected(opened.error());
    auto output = std::move(*opened);
    output.write(format.magic.data(), format.magic.size());
     write_scalar(output, tile_binary_version);
    write_scalar(output, static_cast<std::uint32_t>(tile_edge));
    write_scalar(output, voxel_size);
    write_scalar(output, distance);
    write_scalar(output, static_cast<std::uint64_t>(order.size()));
    if (!output) return std::unexpected(operation_error{"cannot write map tile export"});
    for (const auto index : order) {
        write_scalar(output, static_cast<std::int32_t>(index.x));
        write_scalar(output, static_cast<std::int32_t>(index.y));
        write_scalar(output, static_cast<std::int32_t>(index.z));
        if (!output) return std::unexpected(operation_error{"cannot write map tile export"});
        for (const auto& voxel : tiles.at(index).voxels) {
            const auto written = write_voxel(output, voxel);
            if (!written) return written;
        }
    }
    output.close();
    if (!output) return std::unexpected(operation_error{"cannot close map tile export"});
    return {};
}
}
}

namespace covoxslam
{
operation_status validate_export_submap(const submap& value)
{
    const auto& frame = value.frame;
    const auto& config = value.config;
    const auto& statistics = value.statistics;
    if (!config.valid() || frame.id > maximum_exact_json_integer || frame.frame_ids.empty() ||
        frame.end_stamp_ns <= frame.start_stamp_ns)
        return std::unexpected(operation_error{"invalid submap export descriptor"});
    for (const auto* pose : {&frame.anchor, &value.optimized_world_pose})
        for (const double component : {pose->x, pose->y, pose->z, pose->yaw})
            if (!std::isfinite(component))
                return std::unexpected(operation_error{"invalid exported submap pose"});
    if (!std::isfinite(statistics.integration_ms) || statistics.integration_ms < 0 ||
        !std::isfinite(statistics.esdf_ms) || statistics.esdf_ms < 0 ||
         statistics.rays_rejected > statistics.rays_submitted ||
        statistics.rays_integrated != statistics.rays_submitted - statistics.rays_rejected)
        return std::unexpected(operation_error{"inconsistent exported mapping statistics"});
    auto frame_ids = frame.frame_ids;
    std::sort(frame_ids.begin(), frame_ids.end());
    if (frame_ids.front() < 0 || std::adjacent_find(frame_ids.begin(), frame_ids.end()) != frame_ids.end())
        return std::unexpected(operation_error{"exported frame identifiers must be nonnegative and unique"});
    for (const auto& [block, tile] : value.tsdf)
        for (const auto& voxel : tile.voxels)
        {
            const bool observed = (voxel.flags & observed_flag) != 0;
            if (!std::isfinite(voxel.distance) || !std::isfinite(voxel.weight) || voxel.weight < 0 ||
                voxel.weight > config.maximum_weight || std::abs(voxel.distance) > config.truncation_distance ||
                (voxel.flags & ~(observed_flag | near_surface_flag)) != 0 ||
                 observed != (voxel.weight > 0) || observed != (voxel.sample_count != 0) ||
                (!observed && (voxel.flags & near_surface_flag) != 0))
                return std::unexpected(operation_error{"inconsistent TSDF voxel export"});
        }
    for (const auto& [block, tile] : value.esdf)
    {
        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& voxel = tile.voxels[local];
            const bool observed = (voxel.flags & observed_flag) != 0;
            const bool site = (voxel.flags & esdf_site_flag) != 0;
            const bool signed_seed = site && config.esdf_generator == esdf_backend::cuda_blocks_batch;
            if (signed_seed) {
                const auto source_index = global_index(block, local);
                if (!source_index) return std::unexpected(source_index.error());
                const auto* source = find_voxel(value.tsdf, *source_index);
                if (!source || !(source->flags & observed_flag) ||
                    source->weight < config.esdf_minimum_weight || std::abs(source->distance) >= config.esdf_site_band_m ||
                    std::bit_cast<std::uint32_t>(source->distance) !=
                        std::bit_cast<std::uint32_t>(voxel.signed_distance_m))
                    return std::unexpected(operation_error{"inconsistent blocks fixed seed export"});
            }
            if (!std::isfinite(voxel.signed_distance_m) ||
                std::abs(voxel.signed_distance_m) > config.esdf_maximum_distance_m ||
                (voxel.flags & ~(observed_flag | esdf_site_flag)) != 0 ||
                 (site && (!observed || (!signed_seed && voxel.signed_distance_m != 0) ||
                           voxel.parent_dx != 0 || voxel.parent_dy != 0 || voxel.parent_dz != 0)))
                return std::unexpected(operation_error{"inconsistent ESDF voxel export"});
            const std::array<std::int64_t, 3> index{
                std::int64_t(block.x) * tile_edge + static_cast<int>(local % tile_edge),
                std::int64_t(block.y) * tile_edge + static_cast<int>((local / tile_edge) % tile_edge),
                std::int64_t(block.z) * tile_edge + static_cast<int>(local / (tile_edge * tile_edge))};
            const std::array<std::int32_t, 3> parent{voxel.parent_dx, voxel.parent_dy, voxel.parent_dz};
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const auto source = index[axis] + parent[axis];
                if (index[axis] < std::numeric_limits<int>::min() || index[axis] > std::numeric_limits<int>::max() ||
                    source < std::numeric_limits<int>::min() || source > std::numeric_limits<int>::max())
                    return std::unexpected(operation_error{"ESDF relative parent exceeds the voxel index range"});
            }
        }
    }
    return {};
}
}

namespace covoxslam
{
map_exporter::map_exporter(std::filesystem::path output, export_manifest description,
                           std::ofstream pose_output, std::ofstream metadata_output)
    : directory(std::move(output)), manifest(std::move(description)),
      poses(std::move(pose_output)), metadata(std::move(metadata_output))
{}

operation_result<std::unique_ptr<map_exporter>> map_exporter::create(
    const std::filesystem::path& output, export_manifest description)
{
    if (output.empty() || description.dataset_fingerprint.empty() ||
        description.mapping_backend.empty() || description.esdf_backend.empty())
        return std::unexpected(operation_error{"map export needs an output path and input identity"});
    std::error_code error;
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path(), error);
        if (error) return std::unexpected(operation_error{"cannot create map export parent directory"});
    }
    const auto created = std::filesystem::create_directory(output, error);
    if (error) return std::unexpected(operation_error{"cannot create map export output directory"});
    if (!created) return std::unexpected(operation_error{"map export output already exists"});
    export_directory_cleanup cleanup(output);
    const auto submaps_created = std::filesystem::create_directory(output / submaps_directory_name, error);
    if (error || !submaps_created)
        return std::unexpected(operation_error{"cannot create map export submap directory"});
    auto pose_output = export_stream(output / poses_filename, "cannot open map pose export");
    if (!pose_output) return std::unexpected(pose_output.error());
    auto metadata_output = export_stream(output / submaps_filename, "cannot open map metadata export");
    if (!metadata_output) return std::unexpected(metadata_output.error());
    auto result = std::unique_ptr<map_exporter>(new map_exporter(
        output, std::move(description), std::move(*pose_output), std::move(*metadata_output)));
    cleanup.release();
    return result;
}

operation_status map_exporter::write(const submap& value)
{
    if (finished || failed)
        return std::unexpected(operation_error{"map export is closed or has failed"});
    const auto& frame = value.frame;
    if ((previous_id && frame.id <= *previous_id) || frame.id > maximum_exact_json_integer ||
        frame.frame_ids.empty() || frame.end_stamp_ns <= frame.start_stamp_ns || !value.config.valid())
        return std::unexpected(operation_error{"invalid or unordered submap export"});
    for (const auto* pose : {&frame.anchor, &value.optimized_world_pose})
        for (const double component : {pose->x, pose->y, pose->z, pose->yaw})
            if (!std::isfinite(component))
                return std::unexpected(operation_error{"invalid exported submap pose"});
     if (!std::isfinite(value.statistics.integration_ms) || value.statistics.integration_ms < 0)
        return std::unexpected(operation_error{"invalid exported integration time"});
    const auto valid = validate_export_submap(value);
    if (!valid) return valid;
    failed = true;
    const auto stem = directory / submaps_directory_name /
        (std::string(submap_filename_prefix) + std::to_string(frame.id));
    auto tsdf_path = stem;
    auto esdf_path = stem;
    tsdf_path += tsdf_filename_suffix;
    esdf_path += esdf_filename_suffix;
    const auto tsdf_written = write_tiles(tsdf_path, value.tsdf, tsdf_tile_format,
        value.config.voxel_size, value.config.truncation_distance);
    if (!tsdf_written) return tsdf_written;
    const auto esdf_written = write_tiles(esdf_path, value.esdf, esdf_tile_format,
        value.config.voxel_size, value.config.esdf_maximum_distance_m);
    if (!esdf_written) return esdf_written;
    poses << "{\"submap_id\":" << frame.id << ",\"start_stamp_ns\":" << frame.start_stamp_ns
          << ",\"end_stamp_ns\":" << frame.end_stamp_ns << ",\"initial\":";
    write_pose(poses, frame.anchor);
    poses << ",\"optimized\":";
    write_pose(poses, value.optimized_world_pose);
    poses << "}\n";
    if (!poses) return std::unexpected(operation_error{"cannot write map pose export"});
     metadata << "{\"submap_id\":" << frame.id << ",\"frame_count\":" << frame.frame_ids.size()
             << ",\"start_stamp_ns\":" << frame.start_stamp_ns << ",\"end_stamp_ns\":" << frame.end_stamp_ns
             << ",\"frame_ids\":[";
    for (std::size_t index = 0; index < frame.frame_ids.size(); ++index) {
        if (index != 0) metadata << ',';
        metadata << frame.frame_ids[index];
    }
    const auto& statistics = value.statistics;
    metadata << "],\"mapping_backend\":\"" << json_escape(manifest.mapping_backend)
             << "\",\"fusion_order\":\"" << json_escape(manifest.fusion_order)
             << "\",\"esdf_backend\":\"" << json_escape(manifest.esdf_backend)
             << "\",\"tsdf_tiles\":" << value.tsdf.size() << ",\"esdf_tiles\":" << value.esdf.size();
    write_mapping_parameters(metadata, value.config);
    metadata << ",\"rays_submitted\":" << statistics.rays_submitted
             << ",\"rays_integrated\":" << statistics.rays_integrated
             << ",\"rays_rejected\":" << statistics.rays_rejected
             << ",\"voxel_updates\":" << statistics.voxel_updates
             << ",\"cuda_uploaded_bytes\":" << statistics.cuda_uploaded_bytes
             << ",\"cuda_downloaded_bytes\":" << statistics.cuda_downloaded_bytes
             << ",\"sensor_gather_ms\":" << statistics.sensor_gather_ms
             << ",\"sensor_upload_ms\":" << statistics.sensor_upload_ms
             << ",\"sensor_graph_setup_ms\":" << statistics.sensor_graph_setup_ms
             << ",\"sensor_mask_ms\":" << statistics.sensor_mask_ms
             << ",\"sensor_candidates_ms\":" << statistics.sensor_candidates_ms
             << ",\"sensor_candidate_overflows\":" << statistics.sensor_candidate_overflows
             << ",\"sensor_discovery_ms\":" << statistics.sensor_discovery_ms
             << ",\"sensor_gpu_discovery_ms\":" << statistics.sensor_gpu_discovery_ms
             << ",\"sensor_gpu_allocation_ms\":" << statistics.sensor_gpu_allocation_ms
             << ",\"sensor_resident_dispatch_ms\":" << statistics.sensor_resident_dispatch_ms
             << ",\"sensor_fusion_ms\":" << statistics.sensor_fusion_ms
             << ",\"sensor_readback_ms\":" << statistics.sensor_readback_ms
             << ",\"sensor_candidate_voxels\":" << statistics.sensor_candidate_voxels
             << ",\"sensor_kernel_launches\":" << statistics.sensor_kernel_launches
             << ",\"sensor_transfer_bits\":" << statistics.sensor_transfer_bits
             << ",\"cuda_accumulator_bits\":" << statistics.cuda_accumulator_bits
             << ",\"integration_ms\":" << statistics.integration_ms
             << ",\"payload_bytes\":" << frame.timing.payload_bytes
             << ",\"payload_read_ms\":" << frame.timing.read_ms
             << ",\"payload_validation_ms\":" << frame.timing.validation_ms
             << ",\"payload_decode_ms\":" << frame.timing.decode_ms
             << ",\"deskew_ms\":" << frame.timing.deskew_ms
             << ",\"mapping_callback_ms\":" << frame.timing.integration_ms
             << ",\"buffer_to_map_ms\":" << frame.timing.buffer_to_map_ms()
             << ",\"buffer_to_map_with_readback_ms\":" << frame.timing.buffer_to_map_ms() + statistics.sensor_readback_ms
             << ",\"esdf_ms\":" << statistics.esdf_ms << "}\n";
    if (!metadata) return std::unexpected(operation_error{"cannot write map metadata export"});
    poses.flush();
    if (!poses) return std::unexpected(operation_error{"cannot flush map pose export"});
    metadata.flush();
    if (!metadata) return std::unexpected(operation_error{"cannot flush map metadata export"});
    previous_id = frame.id;
    if (!initial_config) initial_config = value.config;
     ++count;
    failed = false;
    return {};
}

operation_status map_exporter::finish()
{
    if (finished || failed || count == 0)
        return std::unexpected(operation_error{"cannot finalize an empty, closed or failed map export"});
    failed = true;
    poses.close();
    if (!poses) return std::unexpected(operation_error{"cannot close map pose export"});
    metadata.close();
    if (!metadata) return std::unexpected(operation_error{"cannot close map metadata export"});
    const auto temporary = directory / run_pending_filename;
    auto opened = export_stream(temporary, "cannot open map run export");
    if (!opened) return std::unexpected(opened.error());
    auto run = std::move(*opened);
    export_file_cleanup cleanup(temporary);
    run << "{\"schema\":\"" << run_schema << "\",\"dataset_fingerprint\":\""
        << json_escape(manifest.dataset_fingerprint) << "\",\"mapping_backend\":\""
         << json_escape(manifest.mapping_backend) << "\",\"fusion_order\":\"" << json_escape(manifest.fusion_order)
        << "\",\"esdf_backend\":\""
        << json_escape(manifest.esdf_backend) << "\",\"pgo_backend\":\"" << json_escape(manifest.pgo_backend)
        << "\",\"problem_hash\":\"" << json_escape(manifest.problem_hash)
        << "\",\"pgo_publication\":\"" << json_escape(manifest.publication) << "\",\"submap_count\":"
        << count;
    write_mapping_parameters(run, *initial_config);
    run << "}\n";
    if (!run) return std::unexpected(operation_error{"cannot write map run export"});
    run.flush();
    if (!run) return std::unexpected(operation_error{"cannot flush map run export"});
    run.close();
    if (!run) return std::unexpected(operation_error{"cannot close map run export"});
    std::error_code error;
    const auto destination = directory / run_filename;
    if (std::filesystem::exists(destination, error) || error)
        return std::unexpected(operation_error{"cannot publish map export"});
    std::filesystem::rename(temporary, destination, error);
    if (error) return std::unexpected(operation_error{"cannot publish map export"});
    cleanup.release();
    finished = true;
    failed = false;
    return {};
}

operation_status export_run(const std::filesystem::path& output, const export_manifest& manifest,
                            std::span<const submap> submaps)
{
    if (submaps.empty())
        return std::unexpected(operation_error{"cannot export an empty run"});
    std::vector<const submap*> ordered;
    ordered.reserve(submaps.size());
    for (const auto& value : submaps) {
        const auto valid = validate_export_submap(value);
        if (!valid) return valid;
        ordered.push_back(&value);
    }
     std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
        return left->frame.id < right->frame.id;
    });
    for (std::size_t index = 1; index < ordered.size(); ++index)
        if (ordered[index - 1]->frame.id == ordered[index]->frame.id)
            return std::unexpected(operation_error{"duplicate exported submap identifier"});
    auto created = map_exporter::create(output, manifest);
    if (!created) return std::unexpected(created.error());
    for (const auto* value : ordered) {
        const auto written = (*created)->write(*value);
        if (!written) return written;
    }
    return (*created)->finish();
}
}
