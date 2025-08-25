#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/store/geometry_capture.hpp>
#include <covoxslam/mapping/esdf/band_esdf.hpp>
#include <covoxslam/store/band_esdf_generation.hpp>
#include <chrono>

namespace covoxslam::store
{
template<storage::layout l> class band_field_staging
{
    geometry_snapshot<l, storage::schema::esdf> fields_;
    std::uint64_t id_{};
    std::optional<esdf_tiles> values_;
public:
    band_field_staging(geometry_snapshot<l, storage::schema::esdf> fields, std::uint64_t id)
        : fields_(std::move(fields)), id_(id) {}
    operation_result<const esdf_tiles*> values(std::size_t maximum_bytes)
    {
        if (!values_)
        {
            auto converted = geometry_volume<storage::schema::esdf>(fields_, id_, maximum_bytes);
            if (!converted) return std::unexpected(converted.error());
             values_ = std::move(converted->esdf);
        }
        return &*values_;
    }
    const auto& fields() const noexcept { return fields_; }
    operation_result<esdf_tiles> release_values(std::size_t maximum_bytes)
    {
        auto prepared = values(maximum_bytes);
        if (!prepared) return std::unexpected(prepared.error());
        auto result = std::move(*values_);
        values_.reset();
        return result;
    }
    operation_status validate(const coordinated_store<l>& maps) const
    {
        const auto* captured = fields_.find(id_);
        auto current = maps.status(id_);

        if (!current) return std::unexpected(current.error());
        if (!captured || !captured->esdf)
            return std::unexpected(operation_error{"ESDF staging has no retained field"});
        return validate_band_generation(*current, captured->record.versions);
    }
};
template<storage::layout l> operation_result<geometry_snapshot<l, storage::schema::esdf>>
download_band_fields(coordinated_store<l>& maps, std::uint64_t id,
    geometry_parameters parameters, std::size_t maximum_bytes, band_esdf_metrics& metrics)
{
    const auto started = std::chrono::steady_clock::now();
    auto before = maps.status(id);

    if (!before) return std::unexpected(before.error());
    const bool current = before->cpu[1] == before->content[1];
    auto fields = capture_geometry(maps, std::span<const std::uint64_t>(&id, 1), 2,
        parameters, maximum_bytes, current ? std::optional<std::size_t>{} : maximum_bytes);
    if (!fields) return std::unexpected(fields.error());
    const auto* map = fields->find(id);
    if (!map || !map->esdf) return std::unexpected(operation_error{"generated ESDF capture is absent"});

    if (before->cpu[1] != before->content[1])
        metrics.download_bytes += map->esdf->payload.size() * 512 * sizeof(storage::esdf_value);
    metrics.readback_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return fields;
}
}
