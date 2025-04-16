#pragma once
#include <covoxslam/store/handle.hpp>
#include <memory>
#include <mutex>
#include <array>

namespace covoxslam::store
{
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
};
}
