#pragma once
#include <covoxslam/result.hpp>
#include <array>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>

namespace covoxslam::store
{
enum class memory_space : std::size_t { host, pinned, device };
using memory_bytes = std::array<std::size_t, 3>;
struct budget_snapshot
{
    memory_bytes limit{}, used{}, peak{}, pending{};
};
struct budget_state
{
    std::mutex mutex;
    budget_snapshot counters;
};
class budget_claim
{
    std::shared_ptr<budget_state> state_;
    memory_bytes bytes_{};
    bool pending_{};
    friend class memory_budget;
public:
    budget_claim() = default;
    budget_claim(const budget_claim&) = delete;
    budget_claim& operator=(const budget_claim&) = delete;
    ~budget_claim()
    {
        if (!state_) return;
        std::lock_guard lock(state_->mutex);

        for (unsigned i = 0; i < 3; ++i) {
            state_->counters.used[i] -= bytes_[i];
            if (pending_) state_->counters.pending[i] -= bytes_[i];
        }
    }
    void retire()
    {
        if (!state_) return;
        std::lock_guard lock(state_->mutex);
        if (pending_) return;

        for (unsigned i = 0; i < 3; ++i) state_->counters.pending[i] += bytes_[i];
        pending_ = true;
    }
    const memory_bytes& bytes() const noexcept { return bytes_; }
};
class memory_budget
{
    std::shared_ptr<budget_state> state_{std::make_shared<budget_state>()};
public:
    explicit memory_budget(memory_bytes limits)
    {
        state_->counters.limit = limits;
    }
    operation_result<std::shared_ptr<budget_claim>> reserve(memory_bytes bytes)
    {
        auto claim = std::make_shared<budget_claim>();
        std::lock_guard lock(state_->mutex);
        auto& current = state_->counters;

        for (unsigned i = 0; i < 3; ++i)
            if (bytes[i] > current.limit[i] - current.used[i])
                return std::unexpected(operation_error{"map store memory budget exhausted"});

        for (unsigned i = 0; i < 3; ++i) {
            current.used[i] += bytes[i];
            current.peak[i] = std::max(current.peak[i], current.used[i]);
        }
        claim->bytes_ = bytes;
        claim->state_ = state_;
        return claim;
    }
    operation_status configure(memory_bytes limits)
    {
        std::lock_guard lock(state_->mutex);

        for (unsigned i = 0; i < 3; ++i)
            if (limits[i] < state_->counters.used[i]) return std::unexpected(operation_error{"new budget is below retained allocations"});
        state_->counters.limit = limits;
        return {};
    }
    budget_snapshot snapshot() const
    {
        std::lock_guard lock(state_->mutex);
        return state_->counters;
    }
};
inline operation_result<memory_bytes> add_memory(memory_bytes first, memory_bytes second)
{
    for (unsigned i = 0; i < 3; ++i) {
        if (second[i] > std::numeric_limits<std::size_t>::max() - first[i])
            return std::unexpected(operation_error{"memory accounting overflows"});
        first[i] += second[i];
    }
    return first;
}
}
