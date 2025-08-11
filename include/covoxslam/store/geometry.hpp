#pragma once
#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/store/snapshot.hpp>
#include <covoxslam/mapping.hpp>

namespace covoxslam::store
{
enum class geometry_pose { initial, optimized };
struct geometry_parameters
{
    float voxel_size{}, distance_limit{};
    geometry_pose pose{geometry_pose::optimized};
};
inline operation_status validate_geometry(geometry_parameters parameters)
{
    if (!std::isfinite(parameters.voxel_size) || parameters.voxel_size <= 0 ||
        !std::isfinite(parameters.distance_limit) || parameters.distance_limit <= 0 ||
        (parameters.pose != geometry_pose::initial && parameters.pose != geometry_pose::optimized))
        return std::unexpected(operation_error{"geometry parameters are invalid"});
    return {};
}
template<storage::layout l, storage::schema es = storage::schema::esdf> class geometry_snapshot
{
    immutable_snapshot<l, es> snapshot_;
    geometry_parameters parameters_;
    geometry_snapshot(immutable_snapshot<l, es> source, geometry_parameters parameters)
        : snapshot_(std::move(source)), parameters_(parameters) {}
public:
    static operation_result<geometry_snapshot> create(immutable_snapshot<l, es> source, geometry_parameters parameters)
    {
        auto checked = validate_geometry(parameters);
        if (!checked) return std::unexpected(checked.error());
        if (!source.id()) return std::unexpected(operation_error{"geometry snapshot is empty"});
        if (source.maps().empty()) return std::unexpected(operation_error{"geometry snapshot has no submaps"});
        for (const auto& map : source.maps())
            if (!map.record.versions.pose) return std::unexpected(operation_error{"geometry pose epoch is invalid"});
        return geometry_snapshot(std::move(source), parameters);
    }
    std::uint64_t id() const noexcept { return snapshot_.id(); }
    const geometry_parameters& parameters() const noexcept { return parameters_; }
    std::span<const map_snapshot<l, es>> maps() const noexcept { return snapshot_.maps(); }
    const map_snapshot<l, es>* find(std::uint64_t id) const noexcept
    {
        for (const auto& map : snapshot_.maps())
            if (map.record.id == id) return &map;
        return nullptr;
    }
    operation_result<pose_record> pose(std::uint64_t id) const
    {
        const auto* map = find(id);
        if (!map) return std::unexpected(operation_error{"geometry submap is absent"});
        return parameters_.pose == geometry_pose::initial ? map->record.initial_pose : map->record.optimized_pose;
    }
    template<storage::schema s> operation_result<std::optional<storage::value<s>>> voxel(std::uint64_t id, int3 index) const
    {
        const auto* map = find(id);
        if (!map) return std::unexpected(operation_error{"geometry submap is absent"});
        const auto& field = select_field<s, es>(*map);
        if (!field) return std::unexpected(operation_error{"geometry field is not captured"});
        const auto divide = [](int v) { return v / 8 - (v % 8 < 0); };
        const auto local = [](int v) { return unsigned((v % 8 + 8) % 8); };
        const block_key key{id, divide(index.x), divide(index.y), divide(index.z)};
        auto handle = field->index.find(key);
        if (!handle) return std::optional<storage::value<s>>{};
        const auto offset = local(index.x) + 8 * (local(index.y) + 8 * local(index.z));
        return storage::gather_record(field->payload.view()[handle->slot].voxels[offset]);
    }
    operation_result<int3> local_index(std::uint64_t id, double x, double y, double z) const
    {
        auto selected = pose(id);
        if (!selected) return std::unexpected(selected.error());
        const double c = std::cos(selected->yaw), s = std::sin(selected->yaw);
        const double dx = x - selected->x, dy = y - selected->y;
        const double local[]{c * dx + s * dy, -s * dx + c * dy, z - selected->z};
        int values[3];

        for (unsigned axis = 0; axis < 3; ++axis) {
            const auto coordinate = std::floor(local[axis] / parameters_.voxel_size);
            if (!std::isfinite(coordinate) || coordinate < INT32_MIN || coordinate > INT32_MAX)
                return std::unexpected(operation_error{"geometry query exceeds coordinate range"});
            values[axis] = int(coordinate);
        }
        return int3{values[0], values[1], values[2]};
    }
};
}
