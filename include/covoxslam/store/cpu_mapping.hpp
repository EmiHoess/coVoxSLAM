#pragma once
#include <covoxslam/store/coordinator.hpp>
#include <covoxslam/mapping.hpp>

namespace covoxslam::store
{
inline operation_result<int3> mapping_voxel_index(vec3f point, float size)
{
    int3 result;
    int* output[]{&result.x, &result.y, &result.z};
    const float input[]{point.x, point.y, point.z};

    for (unsigned i = 0; i < 3; ++i) {
        const double value = std::floor(double(input[i]) / size);
        if (!std::isfinite(value) || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
            return std::unexpected(operation_error{"mapping voxel coordinate exceeds integer range"});
        *output[i] = int(value);
    }
    return result;
}
template<storage::layout l> operation_result<bool> integrate_store_ray(field_editor<storage::schema::tsdf, l>& editor,
    std::uint64_t id, const ray& input, const mapping_config& config, mapping_statistics& statistics)
{
    const vec3f delta = input.endpoint - input.origin;
    const float distance = length(delta);
    if (!std::isfinite(distance) || distance < config.minimum_ray_length_m || distance > config.maximum_ray_length_m ||
        !std::isfinite(input.weight) || input.weight <= 0 || !std::isfinite(input.origin.x) ||
        !std::isfinite(input.origin.y) || !std::isfinite(input.origin.z)) return false;
    const vec3f direction = delta / distance;
    const double traversal = double(distance) + config.truncation_distance;
    auto start = mapping_voxel_index(input.origin, config.voxel_size);
    auto finish = mapping_voxel_index(input.origin + direction * float(traversal), config.voxel_size);
    if (!start) return std::unexpected(start.error());
    if (!finish) return std::unexpected(finish.error());
    auto voxel = *start;
    std::array<int, 3> step{};
    std::array<double, 3> next{}, increment{};
    const std::array<float, 3> origins{input.origin.x, input.origin.y, input.origin.z};
    const std::array<float, 3> directions{direction.x, direction.y, direction.z};
    const std::array<int, 3> coordinates{voxel.x, voxel.y, voxel.z};

    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (directions[axis] == 0) {
            next[axis] = increment[axis] = std::numeric_limits<double>::infinity();
            continue;
        }
        step[axis] = directions[axis] > 0 ? 1 : -1;
        const double boundary = double(config.voxel_size) * (double(coordinates[axis]) + (step[axis] > 0 ? 1 : 0));
        next[axis] = std::max(0.0, (boundary - origins[axis]) / directions[axis]);
        increment[axis] = double(config.voxel_size) / std::abs(double(directions[axis]));
    }
    const auto maximum_steps = std::size_t(std::ceil(traversal / config.voxel_size) * 3) + 16;
    double entered{};

    for (std::size_t iteration = 0; iteration < maximum_steps; ++iteration) {
        const double boundary = std::min({next[0], next[1], next[2]});
        const auto center = voxel_center(voxel, config.voxel_size);
        const float projected = dot(center - input.origin, direction);
        if (boundary > entered && projected >= 0 && projected <= traversal) {
            const auto block = block_index(voxel);
            if (!editor.find({id, block.x, block.y, block.z}) && editor.size() >= config.maximum_tiles)
                return std::unexpected(operation_error{"CPU mapping tile budget exceeded"});
            auto handle = editor.allocate({id, block.x, block.y, block.z});
            if (!handle) return std::unexpected(handle.error());
            auto target = editor.voxel(*handle, local_index(voxel));
            if (!target) return std::unexpected(target.error());
            const auto sdf = std::clamp(distance - projected, -config.truncation_distance, config.truncation_distance);
            const double combined = double(target->weight) + input.weight;
            target->distance = float((double(target->distance) * target->weight + double(sdf) * input.weight) / combined);
            target->weight = float(std::min(double(config.maximum_weight), combined));
            if (target->sample_count != std::numeric_limits<std::uint32_t>::max()) ++target->sample_count;
            target->flags = observed_flag | (std::abs(target->distance) <= config.extraction_distance ? near_surface_flag : 0);
            ++statistics.voxel_updates;
        }
        if (!std::isfinite(boundary) || boundary >= traversal) return true;
        int* components[]{&voxel.x, &voxel.y, &voxel.z};

        for (std::size_t axis = 0; axis < 3; ++axis) {
            if (next[axis] > boundary + 1.0e-6) continue;
            const auto candidate = std::int64_t(*components[axis]) + step[axis];
            if (candidate < std::numeric_limits<int>::min() || candidate > std::numeric_limits<int>::max())
                return std::unexpected(operation_error{"mapping DDA coordinate exceeds integer range"});
            *components[axis] = int(candidate);
            next[axis] += increment[axis];
        }
        entered = boundary;
    }
    return std::unexpected(operation_error{"mapping DDA traversal exceeded step limit"});
}
template<storage::layout l, storage::schema es> operation_result<mapping_statistics> integrate_cpu_store(
    coordinated_store<l, es>& store, std::uint64_t id, std::span<const ray> rays, const mapping_config& config)
{
    if (!config.valid()) return std::unexpected(operation_error{"invalid CPU mapping configuration"});
    mapping_statistics statistics;
    const auto begin = std::chrono::steady_clock::now();
    auto checked = store.template edit<storage::schema::tsdf>(id, [&](auto& editor) -> operation_status {

        for (const auto& input : rays) {
            ++statistics.rays_submitted;
            auto integrated = integrate_store_ray(editor, id, input, config, statistics);
            if (!integrated) return std::unexpected(integrated.error());
            if (*integrated) ++statistics.rays_integrated;
            else ++statistics.rays_rejected;
        }
        statistics.allocated_tiles = editor.size();
        return {};
    });
    if (!checked) return std::unexpected(checked.error());
    statistics.integration_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    return statistics;
}
template<storage::layout l> operation_result<tsdf_tiles> canonical_tsdf(const field_pool<storage::schema::tsdf, l>& field)
{
    tsdf_tiles result;
    const auto slots = field.index.records();
    const auto view = field.payload.view();

    for (std::size_t i = 0; i < slots.size(); ++i) {
        if (!slots[i].occupied) continue;
        const auto& key = slots[i].key;
        auto& tile = result[int3{key.x, key.y, key.z}];

        for (std::size_t voxel = 0; voxel < 512; ++voxel) {
            auto value = storage::gather_record(view[i].voxels[voxel]);
            tile.voxels[voxel] = {value.distance, value.weight, value.sample_count, value.flags};
        }
    }
    return result;
}
}
