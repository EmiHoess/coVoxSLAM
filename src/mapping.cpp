#include <covoxslam/mapping.hpp>

#include <stdexcept>

namespace covoxslam
{
namespace
{
int floor_divide(int value) noexcept
{
    return value / tile_edge - (value % tile_edge < 0);
}

int floor_modulo(int value) noexcept {
    const int remainder = value % tile_edge;
     return remainder < 0 ? remainder + tile_edge : remainder;
}
}

std::size_t int3_hash::operator()(int3 value) const noexcept
{
    std::uint64_t hash = 1469598103934665603ULL;
    for (const int component : {value.x, value.y, value.z})
    {
        hash ^= static_cast<std::uint32_t>(component);
        hash *= 1099511628211ULL;
    }
    return static_cast<std::size_t>(hash);
}

int3 block_index(int3 voxel) noexcept
{
     return {floor_divide(voxel.x), floor_divide(voxel.y), floor_divide(voxel.z)};
}

std::size_t local_index(int3 voxel) noexcept {
    const int x = floor_modulo(voxel.x);
    const int y = floor_modulo(voxel.y);
    const int z = floor_modulo(voxel.z);
    return static_cast<std::size_t>(x + tile_edge * (y + tile_edge * z));
}

bool block_index_less(int3 left, int3 right) noexcept
{
    if (left.z != right.z) return left.z < right.z;
    if (left.y != right.y) return left.y < right.y;
    return left.x < right.x;
}

const tsdf_voxel* find_voxel(const tsdf_tiles& tiles, int3 voxel) noexcept
{
    const auto found = tiles.find(block_index(voxel));
     return found == tiles.end() ? nullptr : &found->second.voxels[local_index(voxel)];
}

tsdf_voxel& allocate_voxel(tsdf_tiles& tiles, int3 voxel, std::size_t maximum_tiles)
{
    const auto block = block_index(voxel);
    auto found = tiles.find(block);
    if (found == tiles.end()) {
        if (tiles.size() >= maximum_tiles) throw std::runtime_error("TSDF tile budget exceeded");
        found = tiles.emplace(block, tsdf_tile{}).first;
    }
    return found->second.voxels[local_index(voxel)];
}
}

#include <cmath>
#include <limits>

namespace covoxslam
{
int3 voxel_index(vec3f point, float voxel_size)
{
    if (!std::isfinite(voxel_size) || voxel_size <= 0)
        throw std::invalid_argument("voxel size must be finite and positive");
    const auto index = [voxel_size](float coordinate)
    {
        const double value = std::floor(double(coordinate) / voxel_size);
        if (!std::isfinite(value) || value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max())
            throw std::out_of_range("voxel coordinate exceeds integer range");
         return static_cast<int>(value);
    };
    return {index(point.x), index(point.y), index(point.z)};
}

int3 global_index(int3 block, std::size_t local)
{
    if (local >= tile_voxels) throw std::out_of_range("tile voxel index out of bounds");
    const auto coordinate = [](int value, int offset) {
        const auto result = std::int64_t(value) * tile_edge + offset;
        if (result < std::numeric_limits<int>::min() || result > std::numeric_limits<int>::max())
            throw std::out_of_range("block coordinate exceeds voxel range");
        return static_cast<int>(result);
    };
     return {coordinate(block.x, static_cast<int>(local % tile_edge)),
             coordinate(block.y, static_cast<int>((local / tile_edge) % tile_edge)),
             coordinate(block.z, static_cast<int>(local / (tile_edge * tile_edge)))};
}

vec3f voxel_center(int3 index, float size) noexcept
{
    return {static_cast<float>(size * (double(index.x) + 0.5)),
            static_cast<float>(size * (double(index.y) + 0.5)),
            static_cast<float>(size * (double(index.z) + 0.5))};
}
}

namespace covoxslam
{
void cpu_mapper::begin(const submap_frame& frame)
{
    if (active) throw std::logic_error("finish the active submap before beginning another");
    if (frame.id < 0 || frame.end_stamp_ns < frame.start_stamp_ns ||
        !std::isfinite(frame.anchor.x) || !std::isfinite(frame.anchor.y) ||
        !std::isfinite(frame.anchor.z) || !std::isfinite(frame.anchor.yaw))
        throw std::invalid_argument("invalid submap anchor or interval");
    active.emplace();
    active->config = config;
    active->frame = frame;
     active->optimized_world_pose = frame.anchor;
}

submap cpu_mapper::finish(const submap_frame& frame)
{
    if (!active || frame.id != active->frame.id || frame.start_stamp_ns != active->frame.start_stamp_ns ||
        frame.end_stamp_ns < frame.start_stamp_ns)
        throw std::logic_error("submap finalization does not match the active map");
    const auto& anchor = active->frame.anchor;
    if (frame.anchor.x != anchor.x || frame.anchor.y != anchor.y ||
        frame.anchor.z != anchor.z || frame.anchor.yaw != anchor.yaw)
        throw std::logic_error("cannot change the integration anchor of a submap");
    active->frame = frame;
    submap result = std::move(*active);
    active.reset();
    return result;
}

bool cpu_mapper::is_active() const noexcept {
    return active.has_value();
}
}

namespace covoxslam
{
bool mapping_config::valid() const noexcept
{
    for (const auto value : {voxel_size, truncation_distance, extraction_distance,
                            maximum_weight, minimum_ray_length_m, maximum_ray_length_m,
                            esdf_site_band_m, esdf_maximum_distance_m, esdf_minimum_weight})
        if (!std::isfinite(value) || value <= 0) return false;
    return maximum_tiles > 0 && minimum_ray_length_m <= maximum_ray_length_m &&
        extraction_distance <= truncation_distance &&
        esdf_site_band_m <= truncation_distance && esdf_site_band_m <= esdf_maximum_distance_m &&
        double(esdf_maximum_distance_m) / voxel_size < 10000000 &&
        double(maximum_ray_length_m) + truncation_distance <= std::numeric_limits<float>::max() &&
        (double(maximum_ray_length_m) + truncation_distance) / voxel_size < 10000000;
}

cpu_mapper::cpu_mapper(mapping_config options) : config(options)
{
     if (!config.valid()) throw std::invalid_argument("invalid CPU mapping configuration");
}
}

#include <algorithm>
#include <chrono>

namespace covoxslam
{
namespace
{
void update_tsdf(submap& map, int3 index, float distance, float weight)
{
    auto& voxel = allocate_voxel(map.tsdf, index, map.config.maximum_tiles);
    const double combined = double(voxel.weight) + weight;
    voxel.distance = static_cast<float>((double(voxel.distance) * voxel.weight +
                                        double(distance) * weight) / combined);
    voxel.weight = static_cast<float>(std::min(double(map.config.maximum_weight), combined));
    if (voxel.sample_count != std::numeric_limits<std::uint32_t>::max()) ++voxel.sample_count;
     voxel.flags = observed_flag | (std::abs(voxel.distance) <= map.config.extraction_distance ? near_surface_flag : 0);
    ++map.statistics.voxel_updates;
}

bool integrate_ray(submap& map, const ray& input)
{
    const auto& config = map.config;
    const vec3f delta = input.endpoint - input.origin;
    const float distance = length(delta);
    if (!std::isfinite(distance) || distance < config.minimum_ray_length_m ||
        distance > config.maximum_ray_length_m || !std::isfinite(input.weight) || input.weight <= 0 ||
        !std::isfinite(input.origin.x) || !std::isfinite(input.origin.y) || !std::isfinite(input.origin.z))
        return false;
    const vec3f direction = delta / distance;
    const double traversal = double(distance) + config.truncation_distance;
     auto voxel = voxel_index(input.origin, config.voxel_size);
    voxel_index(input.origin + direction * static_cast<float>(traversal), config.voxel_size);
    std::array<int, 3> step{};
    std::array<double, 3> next{}, increment{};
    const std::array<float, 3> origins{input.origin.x, input.origin.y, input.origin.z};
    const std::array<float, 3> directions{direction.x, direction.y, direction.z};
    const std::array<int, 3> coordinates{voxel.x, voxel.y, voxel.z};
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (directions[axis] == 0) {
            next[axis] = increment[axis] = std::numeric_limits<double>::infinity();
            continue;
        }
        step[axis] = directions[axis] > 0 ? 1 : -1;
        const double boundary = double(config.voxel_size) *
            (double(coordinates[axis]) + (step[axis] > 0 ? 1 : 0));
        next[axis] = std::max(0.0, (boundary - origins[axis]) / directions[axis]);
         increment[axis] = double(config.voxel_size) / std::abs(double(directions[axis]));
    }
    const auto maximum_steps = static_cast<std::size_t>(std::ceil(double(traversal) / config.voxel_size) * 3) + 16;
    bool complete{};
    double entered{};
    for (std::size_t iteration = 0; iteration < maximum_steps; ++iteration)
    {
        const double boundary = std::min({next[0], next[1], next[2]});
        const auto center = voxel_center(voxel, config.voxel_size);
        const float projected = dot(center - input.origin, direction);
        if (boundary > entered && projected >= 0 && projected <= traversal)
            update_tsdf(map, voxel, std::clamp(distance - projected,
                        -config.truncation_distance, config.truncation_distance), input.weight);
        if (!std::isfinite(boundary) || boundary >= traversal) {
            complete = true;
            break;
        }
         int* components[]{&voxel.x, &voxel.y, &voxel.z};
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            if (next[axis] > boundary + 1.0e-6) continue;
            const auto candidate = std::int64_t(*components[axis]) + step[axis];
            if (candidate < std::numeric_limits<int>::min() || candidate > std::numeric_limits<int>::max())
                throw std::out_of_range("DDA coordinate exceeds integer range");
            *components[axis] = static_cast<int>(candidate);
            next[axis] += increment[axis];
        }
        entered = boundary;
    }
    if (!complete) throw std::runtime_error("DDA traversal exceeded step limit");
    return true;
}
}

void cpu_mapper::integrate_scan(std::span<const ray> rays)
{
    if (!active) throw std::logic_error("begin a submap before integration");
     const auto start = std::chrono::steady_clock::now();
    auto& statistics = active->statistics;
    for (const auto& input : rays) {
        ++statistics.rays_submitted;
        if (integrate_ray(*active, input)) ++statistics.rays_integrated;
        else ++statistics.rays_rejected;
    }
    statistics.allocated_tiles = active->tsdf.size();
    statistics.integration_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
}
}

#include "mapping_internal.hpp"

namespace covoxslam
{
void build_cpu_esdf(const tsdf_tiles& tsdf, esdf_tiles& esdf, const mapping_config& config)
{
    if (!config.valid()) throw std::invalid_argument("invalid ESDF mapping configuration");
    esdf_tiles output;
    detail::esdf_states states;
     detail::esdf_queue queue;
    for (const auto& [block, tile] : tsdf)
    {
        for (std::size_t local = 0; local < tile_voxels; ++local) {
            const auto& voxel = tile.voxels[local];
            if (!detail::eligible_esdf_voxel(voxel, config)) continue;
            const auto index = global_index(block, local);
            states.emplace(index, detail::esdf_state{});
            auto& destination = output[block].voxels[local];
            destination.flags = observed_flag;
            if (std::abs(voxel.distance) <= config.esdf_site_band_m)
            {
                 states[index] = {0, index};
                queue.push({0, index, index});
                destination.flags |= esdf_site_flag;
            }
        }
    }
    while (!queue.empty())
    {
        const auto current = queue.top();
        queue.pop();
        const auto state = states.find(current.voxel);
        if (state == states.end() || current.distance != state->second.distance) continue;
        for (int z = -1; z <= 1; ++z)
            for (int y = -1; y <= 1; ++y)
                for (int x = -1; x <= 1; ++x)
                {
                    if (x == 0 && y == 0 && z == 0) continue;
                    int3 neighbor;
                    if (!detail::offset_index(current.voxel, x, y, z, neighbor)) continue;
                    const auto found = states.find(neighbor);
                     if (found == states.end()) continue;
                    const double dx = double(neighbor.x) - current.source.x;
                    const double dy = double(neighbor.y) - current.source.y;
                    const double dz = double(neighbor.z) - current.source.z;
                    const float candidate = static_cast<float>(config.voxel_size * std::sqrt(dx * dx + dy * dy + dz * dz));
                    if (candidate < found->second.distance && candidate <= config.esdf_maximum_distance_m) {
                        found->second = {candidate, current.source};
                        queue.push({candidate, neighbor, current.source});
                    }
                }
    }
    for (const auto& [index, state] : states) {
        const auto* source = find_voxel(tsdf, index);
        auto& destination = output.at(block_index(index)).voxels[local_index(index)];
        const bool reached = std::isfinite(state.distance);
         const float distance = reached ? state.distance : config.esdf_maximum_distance_m;
        destination.signed_distance_m = source->distance < 0 ? -distance : distance;
        if (reached)
        {
            destination.parent_dx = static_cast<std::int32_t>(std::int64_t(state.source.x) - index.x);
            destination.parent_dy = static_cast<std::int32_t>(std::int64_t(state.source.y) - index.y);
            destination.parent_dz = static_cast<std::int32_t>(std::int64_t(state.source.z) - index.z);
        }
    }
    esdf.swap(output);
}
}
