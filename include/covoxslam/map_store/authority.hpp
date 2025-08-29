#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/schema.hpp>
#include <covoxslam/map_store/key.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <array>

namespace covoxslam::store
{
enum class field_kind : std::uint32_t { tsdf, esdf };
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
struct layout_version
{
    storage::schema schema{};
    storage::layout layout{};
    std::uint32_t format{1}, voxels_per_tile{512};
    bool operator==(const layout_version&) const = default;
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
    std::array<std::uint64_t, 2> content{}, cpu{}, gpu{};
    std::uint64_t esdf_source{}, lease_generation{};
};
struct authority_state
{
    std::mutex mutex;
    authority_snapshot current;
};
inline operation_status require_current_esdf(const authority_snapshot& state, std::uint32_t fields)
{
    if ((fields & 2U) && (!state.content[0] || !state.content[1] || state.esdf_source != state.content[0]))
        return std::unexpected(operation_error{"ESDF requires regeneration from the current TSDF"});
    return {};
}
class mirror_ticket
{
    std::shared_ptr<authority_state> state_;
    writer_kind destination_{};
    std::uint32_t fields_{};
    std::array<std::uint64_t, 2> versions_{};
    bool complete_{};
    friend class map_authority;
    mirror_ticket(std::shared_ptr<authority_state> state, writer_kind destination, std::uint32_t fields,
        std::array<std::uint64_t, 2> versions) : state_(std::move(state)), destination_(destination), fields_(fields), versions_(versions) {}
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

        for (unsigned field = 0; field < 2; ++field)
            if ((fields_ & (1U << field)) && versions_[field] != current.content[field])
                return std::unexpected(operation_error{"mirror completion is stale"});
        auto checked = commit();
        if (!checked) return checked;
        auto& destination = destination_ == writer_kind::cpu ? current.cpu : current.gpu;

        for (unsigned field = 0; field < 2; ++field)
            if (fields_ & (1U << field)) destination[field] = versions_[field];
        complete_ = true;
        return {};
    }
};
class writer_lease
{
    std::shared_ptr<authority_state> state_;
    writer_kind side_{writer_kind::none};
    std::uint32_t fields_{};
    std::uint64_t generation_{};
    friend class map_authority;
    writer_lease(std::shared_ptr<authority_state> state, writer_kind side, std::uint32_t fields, std::uint64_t generation) :
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

        for (unsigned field = 0; field < 2; ++field)
            if (fields_ & (1U << field)) {
                auto epoch = next_epoch(next[field]);
                if (!epoch) return std::unexpected(epoch.error());
                next[field] = *epoch;
            }
        if ((fields_ & 2U) && esdf_source != next[0])
            return std::unexpected(operation_error{"ESDF source does not match the committed TSDF version"});
        auto checked = commit(next);
        if (!checked) return checked;
        auto& replica = side_ == writer_kind::cpu ? current.cpu : current.gpu;

        for (unsigned field = 0; field < 2; ++field)
            if (fields_ & (1U << field)) replica[field] = next[field];
        current.content = next;
        if (fields_ & 2U) current.esdf_source = esdf_source;
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
            current.content[0] || current.content[1] || record.state != lifecycle::finalized ||
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
    operation_result<writer_lease> acquire(writer_kind side, std::uint32_t fields)
    {
        std::lock_guard lock(state_->mutex);
        auto& current = state_->current;
        if ((side != writer_kind::cpu && side != writer_kind::gpu) || !fields || (fields & ~3U) ||
            current.state != lifecycle::active || current.writer != writer_kind::none)
            return std::unexpected(operation_error{"writer admission conflicts"});
        const auto& replica = side == writer_kind::cpu ? current.cpu : current.gpu;

        for (unsigned field = 0; field < 2; ++field)
            if ((fields & (1U << field)) && replica[field] != current.content[field])
                return std::unexpected(operation_error{"writer replica requires synchronization"});
        auto generation = next_epoch(current.lease_generation);
        if (!generation) return std::unexpected(generation.error());
        current.lease_generation = *generation;
        current.writer = side;
        return writer_lease(state_, side, fields, *generation);
    }
    operation_result<mirror_ticket> transfer(writer_kind source, writer_kind destination, std::uint32_t fields)
    {
        std::lock_guard lock(state_->mutex);
        const auto& current = state_->current;
        if ((source != writer_kind::cpu && source != writer_kind::gpu) ||
            (destination != writer_kind::cpu && destination != writer_kind::gpu) || source == destination ||
            !fields || (fields & ~3U) || current.state == lifecycle::retired || current.writer != writer_kind::none)
            return std::unexpected(operation_error{"mirror transfer admission conflicts"});
        const auto& replica = source == writer_kind::cpu ? current.cpu : current.gpu;

        for (unsigned field = 0; field < 2; ++field)
            if ((fields & (1U << field)) && (!current.content[field] || replica[field] != current.content[field]))
                return std::unexpected(operation_error{"mirror source is not current"});
        return mirror_ticket(state_, destination, fields, current.content);
    }
    template<class function> operation_status finalize(std::uint32_t required_fields, function&& commit)
    {
        std::lock_guard lock(state_->mutex);
        auto& current = state_->current;
        if (!required_fields || (required_fields & ~3U) || current.state != lifecycle::active || current.writer != writer_kind::none)
            return std::unexpected(operation_error{"finalization conflicts"});

        for (unsigned field = 0; field < 2; ++field)
            if ((required_fields & (1U << field)) && (!current.content[field] ||
                (current.cpu[field] != current.content[field] && current.gpu[field] != current.content[field])))
                return std::unexpected(operation_error{"finalization requires complete current fields"});
        if ((required_fields & 2U) && current.esdf_source != current.content[0])
            return std::unexpected(operation_error{"finalization requires current ESDF derivation"});
        auto checked = commit();
        if (!checked) return checked;
        current.state = lifecycle::finalized;
        return {};
    }
    operation_status finalize(std::uint32_t required_fields)
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

        for (unsigned field = 0; field < 2; ++field)
            if (current.content[field] && alternate[field] != current.content[field])
                return std::unexpected(operation_error{"eviction requires a complete recoverable replica"});
        auto checked = remove();
        if (!checked) return checked;
        (side == writer_kind::cpu ? current.cpu : current.gpu) = {};
        return {};
    }
};
}
