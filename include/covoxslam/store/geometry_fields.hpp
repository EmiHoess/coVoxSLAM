#pragma once
#include <covoxslam/store/geometry.hpp>

namespace covoxslam::store
{
struct geometry_volume_data
{
    tsdf_tiles tsdf;
    esdf_tiles esdf;
};
template<storage::schema s> operation_result<esdf_voxel> geometry_esdf(storage::value<s> source,
    int3 position, geometry_parameters parameters)
{
    esdf_voxel result;
    if constexpr (s == storage::schema::esdf)
        result = {source.signed_distance_m, source.parent_dx, source.parent_dy, source.parent_dz, source.flags};
    else if constexpr (s == storage::schema::blocks_esdf) {
        if (source.flags & ~7U) return std::unexpected(operation_error{"invalid signed ESDF flags"});
        result = {source.distance_m, source.parent_dx, source.parent_dy, source.parent_dz, source.flags & 5U};
    }
    else {
        const auto squared = source.squared_distance_voxels;
        const float maximum = parameters.distance_limit / parameters.voxel_size;
        if (!std::isfinite(squared) || squared < 0 || (source.flags & ~15U) || !std::isfinite(maximum * maximum))
            return std::unexpected(operation_error{"invalid squared ESDF value"});
        const bool assigned = source.flags & 8U, site = source.flags & 4U;
        const bool clipped = squared >= maximum * maximum;
        if ((site && (!assigned || !(source.flags & 2U) || squared != 0)) || (!assigned && !clipped))
            return std::unexpected(operation_error{"invalid squared ESDF parent"});
        const auto distance = clipped ? parameters.distance_limit : std::min(parameters.distance_limit, parameters.voxel_size * std::sqrt(squared));
        result.signed_distance_m = std::copysign(distance, source.flags & 2U ? -1.f : 1.f);
        result.flags = source.flags & 5U;
        if (assigned && !clipped) {
            const std::int64_t delta[]{std::int64_t(source.parent_x) - position.x,
                std::int64_t(source.parent_y) - position.y, std::int64_t(source.parent_z) - position.z};

            for (auto value : delta)
                if (value < INT32_MIN || value > INT32_MAX) return std::unexpected(operation_error{"ESDF parent offset overflows"});
            if (site && (delta[0] || delta[1] || delta[2])) return std::unexpected(operation_error{"ESDF site parent differs"});
            result.parent_dx = int(delta[0]); result.parent_dy = int(delta[1]); result.parent_dz = int(delta[2]);
        }
    }
    if (!std::isfinite(result.signed_distance_m) || (result.flags & ~5U) || (!(source.flags & 1U) && (source.flags & ~1U)))
        return std::unexpected(operation_error{"invalid ESDF geometry value"});
    if (std::abs(result.signed_distance_m) >= parameters.distance_limit) {
        result.signed_distance_m = std::copysign(parameters.distance_limit, result.signed_distance_m);
        result.parent_dx = result.parent_dy = result.parent_dz = 0;
    }
    return result;
}
template<storage::schema s, storage::layout l, storage::schema es>
operation_result<std::optional<float>> query_distance(const geometry_snapshot<l, es>& source,
    std::uint64_t id, double x, double y, double z)
{
    auto index = source.local_index(id, x, y, z);
    if (!index) return std::unexpected(index.error());
    auto value = source.template voxel<s>(id, *index);
    if (!value) return std::unexpected(value.error());
    if (!*value || !(**value).flags) return std::optional<float>{};
    if (!((**value).flags & 1U)) return std::optional<float>{};
    if constexpr (s == storage::schema::tsdf) {
        if (!std::isfinite((**value).distance) || !std::isfinite((**value).weight) || (**value).weight <= 0)
            return std::unexpected(operation_error{"invalid TSDF geometry value"});
        return (**value).distance;
    }
    else {
        auto converted = geometry_esdf<s>(**value, *index, source.parameters());
        if (!converted) return std::unexpected(converted.error());
        return converted->signed_distance_m;
    }
}
template<storage::schema s, storage::layout l, storage::schema es>
operation_result<geometry_volume_data> geometry_volume(const geometry_snapshot<l, es>& source,
    std::uint64_t id, std::size_t maximum_payload_bytes)
{
    const auto* map = source.find(id);
    if (!map) return std::unexpected(operation_error{"geometry submap is absent"});
    const auto& field = [&]() -> const auto& {
        if constexpr (s == storage::schema::tsdf) return map->tsdf;
        else return map->esdf;
    }();
    if (!field) return std::unexpected(operation_error{"geometry field is not captured"});
    if (field->index.size() > maximum_payload_bytes / (512 * sizeof(storage::value<s>)))
        return std::unexpected(operation_error{"geometry payload limit exceeded"});
    geometry_volume_data result;
    const auto records = field->index.records();

    for (std::size_t i = 0; i < records.size(); ++i) {
        if (!records[i].occupied) continue;
        const auto key = records[i].key;
        const std::int64_t bases[]{std::int64_t(key.x) * 8, std::int64_t(key.y) * 8, std::int64_t(key.z) * 8};

        for (auto base : bases)
            if (base < INT32_MIN || base + 7 > INT32_MAX) return std::unexpected(operation_error{"geometry tile overflows"});
        std::conditional_t<s == storage::schema::tsdf, tsdf_tile, esdf_tile> tile;

        for (unsigned j = 0; j < 512; ++j) {
            const auto value = storage::gather_record(field->payload.view()[i].voxels[j]);
            if constexpr (s == storage::schema::tsdf) {
                if (!std::isfinite(value.distance) || !std::isfinite(value.weight) || value.weight < 0 || (value.flags & ~3U))
                    return std::unexpected(operation_error{"invalid TSDF geometry value"});
                tile.voxels[j] = {value.distance, value.weight, value.sample_count, value.flags};
            }
            else {
                const int3 position{int(bases[0] + (j & 7)), int(bases[1] + ((j >> 3) & 7)), int(bases[2] + (j >> 6))};
                auto converted = geometry_esdf<s>(value, position, source.parameters());
                if (!converted) return std::unexpected(converted.error());
                tile.voxels[j] = *converted;
            }
        }
        if constexpr (s == storage::schema::tsdf) result.tsdf.emplace(int3{key.x, key.y, key.z}, std::move(tile));
        else result.esdf.emplace(int3{key.x, key.y, key.z}, std::move(tile));
    }
    return result;
}
}
