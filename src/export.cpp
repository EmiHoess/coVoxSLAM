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

namespace covoxslam
{
namespace
{
void validate_export_submap(const submap& value)
{
    const auto& frame = value.frame;
    const auto& config = value.config;
    const auto& statistics = value.statistics;
    if (!config.valid() || frame.id > 9007199254740991ULL || frame.frame_ids.empty() ||
        frame.end_stamp_ns <= frame.start_stamp_ns)
        throw std::invalid_argument("invalid submap export descriptor");
    for (const auto* pose : {&frame.anchor, &value.optimized_world_pose})
        for (const double component : {pose->x, pose->y, pose->z, pose->yaw})
            if (!std::isfinite(component)) throw std::invalid_argument("invalid exported submap pose");
    if (!std::isfinite(statistics.integration_ms) || statistics.integration_ms < 0 ||
        !std::isfinite(statistics.esdf_ms) || statistics.esdf_ms < 0 ||
         statistics.rays_rejected > statistics.rays_submitted ||
        statistics.rays_integrated != statistics.rays_submitted - statistics.rays_rejected)
        throw std::invalid_argument("inconsistent exported mapping statistics");
    auto frame_ids = frame.frame_ids;
    std::sort(frame_ids.begin(), frame_ids.end());
    if (frame_ids.front() < 0 || std::adjacent_find(frame_ids.begin(), frame_ids.end()) != frame_ids.end())
        throw std::invalid_argument("exported frame identifiers must be nonnegative and unique");
    for (const auto& [block, tile] : value.tsdf)
        for (const auto& voxel : tile.voxels)
        {
            const bool observed = (voxel.flags & observed_flag) != 0;
            if (!std::isfinite(voxel.distance) || !std::isfinite(voxel.weight) || voxel.weight < 0 ||
                voxel.weight > config.maximum_weight || std::abs(voxel.distance) > config.truncation_distance ||
                (voxel.flags & ~(observed_flag | near_surface_flag)) != 0 ||
                 observed != (voxel.weight > 0) || observed != (voxel.sample_count != 0) ||
                (!observed && (voxel.flags & near_surface_flag) != 0))
                throw std::invalid_argument("inconsistent TSDF voxel export");
        }
    for (const auto& [block, tile] : value.esdf)
    {
        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& voxel = tile.voxels[local];
            const bool observed = (voxel.flags & observed_flag) != 0;
            const bool site = (voxel.flags & esdf_site_flag) != 0;
            const bool signed_seed = site && config.esdf_generator == esdf_backend::cuda_blocks_batch;
            if (signed_seed) {
                const auto* source = find_voxel(value.tsdf, global_index(block, local));
                if (!source || !(source->flags & observed_flag) ||
                    source->weight < config.esdf_minimum_weight || std::abs(source->distance) >= config.esdf_site_band_m ||
                    std::bit_cast<std::uint32_t>(source->distance) !=
                        std::bit_cast<std::uint32_t>(voxel.signed_distance_m))
                    throw std::invalid_argument("inconsistent blocks fixed seed export");
            }
            if (!std::isfinite(voxel.signed_distance_m) ||
                std::abs(voxel.signed_distance_m) > config.esdf_maximum_distance_m ||
                (voxel.flags & ~(observed_flag | esdf_site_flag)) != 0 ||
                 (site && (!observed || (!signed_seed && voxel.signed_distance_m != 0) ||
                           voxel.parent_dx != 0 || voxel.parent_dy != 0 || voxel.parent_dz != 0)))
                throw std::invalid_argument("inconsistent ESDF voxel export");
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
                    throw std::out_of_range("ESDF relative parent exceeds the voxel index range");
            }
        }
    }
}
}
}

namespace covoxslam
{
map_exporter::map_exporter(const std::filesystem::path& output, export_manifest description)
    : directory(output), manifest(std::move(description))
{
    if (directory.empty() || manifest.dataset_fingerprint.empty() ||
        manifest.mapping_backend.empty() || manifest.esdf_backend.empty())
        throw std::invalid_argument("map export needs an output path and input identity");
    if (!directory.parent_path().empty()) std::filesystem::create_directories(directory.parent_path());
     if (!std::filesystem::create_directory(directory))
        throw std::invalid_argument("map export output already exists");
    std::filesystem::create_directory(directory / "submaps");
    poses = export_stream(directory / "poses.jsonl");
    metadata = export_stream(directory / "submaps.jsonl");
}

void map_exporter::write(const submap& value)
{
    if (finished || failed) throw std::logic_error("map export is closed or has failed");
    const auto& frame = value.frame;
    if ((previous_id && frame.id <= *previous_id) || frame.id > 9007199254740991ULL ||
        frame.frame_ids.empty() || frame.end_stamp_ns <= frame.start_stamp_ns || !value.config.valid())
        throw std::invalid_argument("invalid or unordered submap export");
    for (const auto* pose : {&frame.anchor, &value.optimized_world_pose})
        for (const double component : {pose->x, pose->y, pose->z, pose->yaw})
            if (!std::isfinite(component)) throw std::invalid_argument("invalid exported submap pose");
     if (!std::isfinite(value.statistics.integration_ms) || value.statistics.integration_ms < 0)
        throw std::invalid_argument("invalid exported integration time");
    validate_export_submap(value);
    failed = true;
    const auto stem = directory / "submaps" / ("submap_" + std::to_string(frame.id));
    auto tsdf_path = stem;
    auto esdf_path = stem;
    tsdf_path += ".tsdf.bin";
    esdf_path += ".esdf.bin";
    write_tiles(tsdf_path, value.tsdf, "CVXTSDF1",
                value.config.voxel_size, value.config.truncation_distance);
    write_tiles(esdf_path, value.esdf, "CVXESDF1",
                value.config.voxel_size, value.config.esdf_maximum_distance_m);
    poses << "{\"submap_id\":" << frame.id << ",\"start_stamp_ns\":" << frame.start_stamp_ns
          << ",\"end_stamp_ns\":" << frame.end_stamp_ns << ",\"initial\":";
    write_pose(poses, frame.anchor);
    poses << ",\"optimized\":";
    write_pose(poses, value.optimized_world_pose);
    poses << "}\n";
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
    poses.flush();
    metadata.flush();
    previous_id = frame.id;
    if (!initial_config) initial_config = value.config;
     ++count;
    failed = false;
}

void map_exporter::finish()
{
    if (finished || failed || count == 0)
        throw std::logic_error("cannot finalize an empty, closed or failed map export");
    failed = true;
    poses.close();
    metadata.close();
    const auto temporary = directory / "run.json.pending";
    auto run = export_stream(temporary);
    run << "{\"schema\":\"covoxslam.run.v1\",\"dataset_fingerprint\":\""
        << json_escape(manifest.dataset_fingerprint) << "\",\"mapping_backend\":\""
         << json_escape(manifest.mapping_backend) << "\",\"fusion_order\":\"" << json_escape(manifest.fusion_order)
        << "\",\"esdf_backend\":\""
        << json_escape(manifest.esdf_backend) << "\",\"pgo_backend\":\"" << json_escape(manifest.pgo_backend)
        << "\",\"problem_hash\":\"" << json_escape(manifest.problem_hash)
        << "\",\"pgo_publication\":\"" << json_escape(manifest.publication) << "\",\"submap_count\":"
        << count;
    write_mapping_parameters(run, *initial_config);
    run << "}\n";
    run.close();
    std::filesystem::rename(temporary, directory / "run.json");
    finished = true;
    failed = false;
}

void export_run(const std::filesystem::path& output, const export_manifest& manifest,
                 std::span<const submap> submaps)
{
    if (submaps.empty()) throw std::invalid_argument("cannot export an empty run");
    std::vector<const submap*> ordered;
    ordered.reserve(submaps.size());
    for (const auto& value : submaps) {
        validate_export_submap(value);
        ordered.push_back(&value);
    }
     std::sort(ordered.begin(), ordered.end(), [](const auto* left, const auto* right) {
        return left->frame.id < right->frame.id;
    });
    for (std::size_t index = 1; index < ordered.size(); ++index)
        if (ordered[index - 1]->frame.id == ordered[index]->frame.id)
            throw std::invalid_argument("duplicate exported submap identifier");
    map_exporter writer(output, manifest);
    for (const auto* value : ordered) writer.write(*value);
    writer.finish();
}
}
