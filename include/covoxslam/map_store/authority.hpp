#pragma once
#include <covoxslam/core/result.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

namespace covoxslam::store
{

enum class field_kind : std::uint32_t { tsdf, esdf };
enum class field_mask : std::uint32_t { none, tsdf = 1, esdf = 2, all = 3 };

static_assert(static_cast<std::uint32_t>(field_kind::tsdf) == 0 &&
    static_cast<std::uint32_t>(field_kind::esdf) == 1);

static_assert(static_cast<std::uint32_t>(field_mask::none) == 0 &&
    static_cast<std::uint32_t>(field_mask::tsdf) == 1 &&
    static_cast<std::uint32_t>(field_mask::esdf) == 2 &&
    static_cast<std::uint32_t>(field_mask::all) == 3);

    constexpr std::uint32_t field_mask_value(field_mask fields) noexcept
{
    return static_cast<std::uint32_t>(fields);
}

constexpr bool valid_field_mask(field_mask fields) noexcept
{
    return !(field_mask_value(fields) & ~field_mask_value(field_mask::all));
}

constexpr bool any_field(field_mask fields) noexcept
{
    return field_mask_value(fields) != 0;
}

constexpr field_mask field_mask_for(field_kind field) noexcept
{
    if (field == field_kind::tsdf) return field_mask::tsdf;
    if (field == field_kind::esdf) return field_mask::esdf;

    return field_mask::none;
}

constexpr bool has_field(field_mask fields, field_kind field) noexcept
{
    return (field_mask_value(fields) & field_mask_value(field_mask_for(field))) != 0;
}
constexpr field_mask make_field_mask(bool tsdf, bool esdf) noexcept
{
    return static_cast<field_mask>((tsdf ? field_mask_value(field_mask::tsdf) : 0) |
        (esdf ? field_mask_value(field_mask::esdf) : 0));
}

enum class lifecycle : std::uint32_t { active, finalized, retired };
enum class residency : std::uint32_t { absent, resident, evicting, evicted, restoring };
enum class writer_kind : std::uint32_t { none, cpu, gpu };

struct tile_handle
{
    std::uint64_t submap{};
    field_kind field{};
    std::uint32_t slot{std::numeric_limits<std::uint32_t>::max()};
    std::uint64_t reuse_generation{};
    bool operator==(const tile_handle&) const = default;
};

struct epochs {
    std::uint64_t descriptor{1}, allocation{1}, tsdf{}, esdf{}, esdf_source{}, pose{1};
};

struct field_versions
{
    std::uint64_t tsdf{}, esdf{};
    bool operator==(const field_versions&) const = default;
};

struct pose_record
{
    double x{}, y{}, z{}, yaw{};
    bool operator==(const pose_record&) const = default;
};

struct submap_record {

    std::uint64_t id{};
    lifecycle state{lifecycle::active};
    residency cpu{residency::absent}, gpu{residency::absent};
    writer_kind writer{writer_kind::none};
    epochs versions;
    pose_record initial_pose, optimized_pose;
};
inline operation_result<std::uint64_t> next_epoch(std::uint64_t current)
{
    if (current == std::numeric_limits<std::uint64_t>::max())
        return std::unexpected(operation_error{"store epoch is exhausted"});
    return current + 1;
}
struct authority_snapshot
{
    lifecycle state{lifecycle::active};
    writer_kind writer{writer_kind::none};
    field_versions content, cpu, gpu;
    std::uint64_t esdf_source{}, lease_generation{};
};
struct authority_state
{
    std::mutex mutex;
    authority_snapshot current;
};
inline operation_status require_current_esdf(const authority_snapshot& state, field_mask fields)
{
    if (!valid_field_mask(fields)) return std::unexpected(operation_error{"field mask is invalid"});
    if (has_field(fields, field_kind::esdf) &&
        (!state.content.tsdf || !state.content.esdf || state.esdf_source != state.content.tsdf))
        return std::unexpected(operation_error{"ESDF requires regeneration from the current TSDF"});
    return {};
}

class mirror_ticket
{
    std::shared_ptr<authority_state> state_;
    writer_kind destination_{};
    field_mask fields_{field_mask::none};
    field_versions versions_;
    bool complete_{};
    friend class map_authority;
    mirror_ticket(std::shared_ptr<authority_state> state, writer_kind destination, field_mask fields,
        field_versions versions) : state_(std::move(state)), destination_(destination), fields_(fields), versions_(versions) {}
public:
    mirror_ticket(const mirror_ticket&) = delete;
    mirror_ticket& operator=(const mirror_ticket&) = delete;
    mirror_ticket(mirror_ticket&&) noexcept = default;
    mirror_ticket& operator=(mirror_ticket&&) noexcept = default;

    const auto& versions() const noexcept { return versions_; }
    
    template<class function> operation_status acknowledge(function&& commit)
    {
        if (!state_ || complete_) return std::unexpected(operation_error{"mirror completion is not pending"});
        std::lock_guard lock(state_->mutex);
        auto& current = state_->current;
        if (current.state == lifecycle::retired || current.writer != writer_kind::none)
            return std::unexpected(operation_error{"mirror completion conflicts with map state"});

        if ((has_field(fields_, field_kind::tsdf) && versions_.tsdf != current.content.tsdf) ||
            (has_field(fields_, field_kind::esdf) && versions_.esdf != current.content.esdf))
            return std::unexpected(operation_error{"mirror completion is stale"});
        auto checked = commit();
        if (!checked) return checked;
        auto& destination = destination_ == writer_kind::cpu ? current.cpu : current.gpu;

        if (has_field(fields_, field_kind::tsdf)) destination.tsdf = versions_.tsdf;
        if (has_field(fields_, field_kind::esdf)) destination.esdf = versions_.esdf;
        complete_ = true;
        return {};
    }
};

class writer_lease
{
    std::shared_ptr<authority_state> state_;
    writer_kind side_{writer_kind::none};
    field_mask fields_{field_mask::none};
    std::uint64_t generation_{};
    friend class map_authority;

    writer_lease(std::shared_ptr<authority_state> state, writer_kind side, field_mask fields, std::uint64_t generation) :
        state_(std::move(state)), side_(side), fields_(fields), generation_(generation) {}
    void release() noexcept
    {
        if (!state_) return;
        std::lock_guard lock(state_->mutex);
        if (state_->current.lease_generation == generation_) state_->current.writer = writer_kind::none;
    }

public:
    writer_lease(const writer_lease&) = delete;
    writer_lease& operator=(const writer_lease&) = delete;
    writer_lease(writer_lease&& other) noexcept : state_(std::move(other.state_)), side_(other.side_), fields_(other.fields_), generation_(other.generation_) {}
    
    writer_lease& operator=(writer_lease&& other) noexcept
    {
        if (this != &other) {
            release();
            state_ = std::move(other.state_);
            side_ = other.side_; fields_ = other.fields_; generation_ = other.generation_;
        }
        return *this;
    }
    ~writer_lease() { release(); }
    template<class function> operation_status publish(std::uint64_t esdf_source, function&& commit)
    {
        if (!state_) return std::unexpected(operation_error{"writer lease is empty"});
        std::lock_guard lock(state_->mutex);
        auto& current = state_->current;
        if (current.writer != side_ || current.lease_generation != generation_ || current.state != lifecycle::active)
            return std::unexpected(operation_error{"writer lease is stale"});
        auto next = current.content;
        if (has_field(fields_, field_kind::tsdf)) {
            auto epoch = next_epoch(next.tsdf);
            if (!epoch) return std::unexpected(epoch.error());
            next.tsdf = *epoch;
        }
        if (has_field(fields_, field_kind::esdf)) {
            auto epoch = next_epoch(next.esdf);
            if (!epoch) return std::unexpected(epoch.error());
            next.esdf = *epoch;
        }
        if (has_field(fields_, field_kind::esdf) && esdf_source != next.tsdf)
            return std::unexpected(operation_error{"ESDF source does not match the committed TSDF version"});
        auto checked = commit(next);
        if (!checked) return checked;
        auto& replica = side_ == writer_kind::cpu ? current.cpu : current.gpu;

        if (has_field(fields_, field_kind::tsdf)) replica.tsdf = next.tsdf;
        if (has_field(fields_, field_kind::esdf)) replica.esdf = next.esdf;
        current.content = next;
        if (has_field(fields_, field_kind::esdf)) current.esdf_source = esdf_source;
        current.writer = writer_kind::none;
        generation_ = 0;
        return {};
    }
};
class map_authority
{
    std::shared_ptr<authority_state> state_{std::make_shared<authority_state>()};
public:
    operation_status restore(const submap_record& record)
    {
        std::lock_guard lock(state_->mutex);
        const auto& current = state_->current;
        if (current.state != lifecycle::active || current.writer != writer_kind::none || current.lease_generation ||
            current.content.tsdf || current.content.esdf || record.state != lifecycle::finalized ||
            record.writer != writer_kind::none || record.cpu != residency::resident ||
            (!record.versions.tsdf && !record.versions.esdf) ||
            (record.versions.esdf && (!record.versions.tsdf || record.versions.esdf_source != record.versions.tsdf)))
            return std::unexpected(operation_error{"restored authority conflicts"});
        authority_snapshot replacement;
        replacement.state = lifecycle::finalized;
        replacement.content = {record.versions.tsdf, record.versions.esdf};
        replacement.cpu = replacement.content;
        replacement.esdf_source = record.versions.esdf_source;
        state_->current = replacement;
        return {};
    }
    authority_snapshot snapshot() const
    {
        std::lock_guard lock(state_->mutex);
        return state_->current;
    }
    operation_result<writer_lease> acquire(writer_kind side, field_mask fields)
    {
        std::lock_guard lock(state_->mutex);
        auto& current = state_->current;
        if ((side != writer_kind::cpu && side != writer_kind::gpu) || !valid_field_mask(fields) || !any_field(fields) ||
            current.state != lifecycle::active || current.writer != writer_kind::none)
            return std::unexpected(operation_error{"writer admission conflicts"});
        const auto& replica = side == writer_kind::cpu ? current.cpu : current.gpu;

        if ((has_field(fields, field_kind::tsdf) && replica.tsdf != current.content.tsdf) ||
            (has_field(fields, field_kind::esdf) && replica.esdf != current.content.esdf))
            return std::unexpected(operation_error{"writer replica requires synchronization"});
        auto generation = next_epoch(current.lease_generation);
        if (!generation) return std::unexpected(generation.error());
        current.lease_generation = *generation;
        current.writer = side;
        return writer_lease(state_, side, fields, *generation);
    }
    operation_result<mirror_ticket> transfer(writer_kind source, writer_kind destination, field_mask fields)
    {
        std::lock_guard lock(state_->mutex);
        const auto& current = state_->current;
        if ((source != writer_kind::cpu && source != writer_kind::gpu) ||
            (destination != writer_kind::cpu && destination != writer_kind::gpu) || source == destination ||
            !valid_field_mask(fields) || !any_field(fields) || current.state == lifecycle::retired || current.writer != writer_kind::none)
            return std::unexpected(operation_error{"mirror transfer admission conflicts"});
        const auto& replica = source == writer_kind::cpu ? current.cpu : current.gpu;

        if ((has_field(fields, field_kind::tsdf) &&
                (!current.content.tsdf || replica.tsdf != current.content.tsdf)) ||
            (has_field(fields, field_kind::esdf) &&
                (!current.content.esdf || replica.esdf != current.content.esdf)))
            return std::unexpected(operation_error{"mirror source is not current"});
        return mirror_ticket(state_, destination, fields, current.content);
    }
    template<class function> operation_status finalize(field_mask required_fields, function&& commit)
    {
        std::lock_guard lock(state_->mutex);
        auto& current = state_->current;
        if (!valid_field_mask(required_fields) || !any_field(required_fields) ||
            current.state != lifecycle::active || current.writer != writer_kind::none)
            return std::unexpected(operation_error{"finalization conflicts"});
        if ((has_field(required_fields, field_kind::tsdf) && (!current.content.tsdf ||
                (current.cpu.tsdf != current.content.tsdf && current.gpu.tsdf != current.content.tsdf))) ||
            (has_field(required_fields, field_kind::esdf) && (!current.content.esdf ||
                (current.cpu.esdf != current.content.esdf && current.gpu.esdf != current.content.esdf))))
            return std::unexpected(operation_error{"finalization requires complete current fields"});
        if (has_field(required_fields, field_kind::esdf) && current.esdf_source != current.content.tsdf)
            return std::unexpected(operation_error{"finalization requires current ESDF derivation"});
        auto checked = commit();
        if (!checked) return checked;
        current.state = lifecycle::finalized;
        return {};
    }
    operation_status finalize(field_mask required_fields)
    {
        return finalize(required_fields, []() -> operation_status { return {}; });
    }
    operation_status retire()
    {
        std::lock_guard lock(state_->mutex);
        if (state_->current.writer != writer_kind::none || state_->current.state == lifecycle::retired)
            return std::unexpected(operation_error{"retirement conflicts"});
        state_->current.state = lifecycle::retired;
        
        return {};
    }
    template<class function> operation_status evict(writer_kind side, function&& remove)
    {
        std::lock_guard lock(state_->mutex);
        auto& current = state_->current;
        
        if ((side != writer_kind::cpu && side != writer_kind::gpu) || current.state != lifecycle::finalized ||
            current.writer != writer_kind::none)
            return std::unexpected(operation_error{"eviction requires finalized quiescent storage"});
        const auto& alternate = side == writer_kind::cpu ? current.gpu : current.cpu;

        if ((current.content.tsdf && alternate.tsdf != current.content.tsdf) ||
            (current.content.esdf && alternate.esdf != current.content.esdf))
            return std::unexpected(operation_error{"eviction requires a complete recoverable replica"});
        auto checked = remove();

        if (!checked) return checked;
        (side == writer_kind::cpu ? current.cpu : current.gpu) = {};
        return {};
    }
};
}
