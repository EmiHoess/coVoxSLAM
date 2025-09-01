#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/optimization/solver.hpp>
#include <covoxslam/store/pgo_cache_key.hpp>
#include <chrono>
#include <mutex>
#include <ostream>
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
inline operation_status write_preparation_metrics(std::ostream& output, const preparation_metrics& metrics,
    const ceronoceres::SolverTelemetry& solver)
{
    output << "preparation cache hits " << metrics.hits << ", misses " << metrics.misses
        << ", invalidations " << metrics.invalidations << ", key bytes " << metrics.key_bytes
        << ", key ms " << metrics.key_ms << ", prepare ms " << metrics.preparation_ms << '\n'
        << "solver upload bytes " << solver.host_to_device_bytes << ", download bytes " << solver.device_to_host_bytes
        << ", upload ms " << solver.upload_ms << ", total ms " << solver.total_ms << '\n';
    if (!output) return std::unexpected(operation_error{"preparation report write failed"});
    return {};
}
inline operation_result<ceronoceres::SolveResult> solve_prepared_fields(const cached_preparation& prepared,
    const pgo::options& execution, const ceronoceres::NativeLmOptions& lm = {})
{
    switch (execution.solver) {
    case pgo::backend::native_cpu: return pgo::solve_cpu(prepared.problem, execution, lm);
    case pgo::backend::cuda_cold: {
        auto configured = pgo::cuda_solve_options(execution, lm);
        if (!configured) return std::unexpected(configured.error());
        auto solved = ceronoceres::solve_cuda_checked(prepared.problem, configured->common,
            std::get<ceronoceres::NativeCudaColdOptions>(configured->backend_options));
        if (const auto* error = std::get_if<ceronoceres::cuda_solve_error>(&solved))
            return std::unexpected(operation_error{error->message});
        auto result = std::get<ceronoceres::SolveResult>(std::move(solved));
        auto valid = pgo::validate_solve_result(result, prepared.problem.node_count(), ceronoceres::Backend::NativeCudaCold);
        if (!valid) return std::unexpected(valid.error());
        return result;
    }
    default: return std::unexpected(operation_error{"unsupported field solver"});
    }
}
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
