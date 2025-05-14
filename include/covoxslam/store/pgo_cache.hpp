#pragma once
#include <covoxslam/store/pgo_cache_key.hpp>
#include <chrono>
#include <mutex>
#if defined(covoxslam_has_pgo)
namespace covoxslam::store
{
struct preparation_metrics
{
    std::size_t hits{}, misses{}, invalidations{}, key_bytes{};
    double key_ms{}, preparation_ms{};
};
struct cached_preparation
{
    ceronoceres::PreparedProblem problem;
    bool reused{};
};
class preparation_cache
{
    mutable std::mutex mutex_;
    std::optional<preparation_key> key_;
    ceronoceres::PreparedProblem problem_;
    preparation_metrics metrics_;
    std::size_t maximum_key_bytes_;
public:
    explicit preparation_cache(std::size_t maximum_key_bytes = 64 * 1024 * 1024)
        : maximum_key_bytes_(maximum_key_bytes) {}
    preparation_metrics metrics() const
    {
        std::lock_guard lock(mutex_);
        return metrics_;
    }
    void invalidate()
    {
        std::lock_guard lock(mutex_);
        key_.reset();
        problem_ = {};
        metrics_.key_bytes = 0;
        ++metrics_.invalidations;
    }
    operation_result<cached_preparation> prepare(pgo::graph_snapshot graph,
        std::shared_ptr<const ceronoceres::field_snapshot> fields)
    {
        const auto started = std::chrono::steady_clock::now();
        auto checked = pgo::validate_graph(graph);
        if (!checked) return std::unexpected(checked.error());
        if (!fields) return std::unexpected(operation_error{"cache field snapshot is absent"});
        const auto error = ceronoceres::validate_fields(graph.problem, *fields);
        if (!error.empty()) return std::unexpected(operation_error{error});
        auto key = make_preparation_key(graph, *fields, maximum_key_bytes_);
        if (!key) return std::unexpected(key.error());
        const auto generated = std::chrono::steady_clock::now();
        std::lock_guard lock(mutex_);
        metrics_.key_ms += std::chrono::duration<double, std::milli>(generated - started).count();

        if (key_ && *key_ == *key) {
            ++metrics_.hits;
            return cached_preparation{problem_, true};
        }
        auto prepared = ceronoceres::prepare_fields(std::move(graph.problem), std::move(fields));
        if (const auto* failure = std::get_if<std::string>(&prepared))
            return std::unexpected(operation_error{*failure});
        problem_ = std::get<ceronoceres::PreparedProblem>(std::move(prepared));
        metrics_.preparation_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - generated).count();
        ++metrics_.misses;
        metrics_.key_bytes = key->bytes.size();
        key_ = std::move(*key);
        return cached_preparation{problem_, false};
    }
};
}
#endif
