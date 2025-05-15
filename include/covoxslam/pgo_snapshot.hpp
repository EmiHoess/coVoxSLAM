#pragma once
#include <covoxslam/pgo.hpp>

#if defined(covoxslam_has_pgo)
namespace covoxslam::pgo
{
struct snapshot_extents
{
    std::size_t nodes{}, relative_factors{}, absolute_factors{}, groups{}, samples{};
    std::size_t tsdf_tiles{}, esdf_tiles{};
     std::uint64_t record_bytes{};
};
class solve_snapshot
{
    std::shared_ptr<const graph_snapshot> input_;
    ceronoceres::PreparedProblem prepared_;
    snapshot_extents extents_;
    solve_snapshot(std::shared_ptr<const graph_snapshot> input,
        ceronoceres::PreparedProblem prepared, snapshot_extents extents)
        : input_(std::move(input)), prepared_(std::move(prepared)), extents_(extents) {}
    friend operation_result<solve_snapshot> freeze_graph(graph_snapshot graph);
    friend operation_result<solve_snapshot> freeze_field_graph(graph_snapshot,
        std::shared_ptr<const ceronoceres::field_snapshot>);
public:
    const graph_snapshot& graph() const noexcept { return *input_; }
    const ceronoceres::PreparedProblem& prepared() const noexcept { return prepared_; }
    const snapshot_extents& extents() const noexcept { return extents_; }
    const std::string& hash() const noexcept { return prepared_.hash(); }
};
operation_result<snapshot_extents> measure_snapshot(const graph_snapshot& graph);
operation_result<solve_snapshot> freeze_graph(graph_snapshot graph);
operation_result<solve_snapshot> freeze_field_graph(graph_snapshot graph,
    std::shared_ptr<const ceronoceres::field_snapshot> fields);
operation_result<ceronoceres::SolveOptions> cuda_solve_options(
    const options& options, const ceronoceres::NativeLmOptions& lm = {});
operation_result<ceronoceres::SolverTelemetry> cuda_telemetry(const ceronoceres::SolveResult& result);
operation_result<ceronoceres::SolveResult> solve_cuda(const solve_snapshot& snapshot,
    const options& options, const ceronoceres::NativeLmOptions& lm = {});
operation_result<ceronoceres::SolveResult> solve_native(const solve_snapshot& snapshot,
    const options& options, const ceronoceres::NativeLmOptions& lm = {});
operation_status validate_snapshot_result(const solve_snapshot& snapshot, const ceronoceres::SolveResult& result);
struct optimization_options
{
    options execution;
    candidate_options candidates;
    registration_options registration;
    ceronoceres::NativeLmOptions lm;
};
struct optimization_result
{
    solve_snapshot snapshot;
    ceronoceres::SolveResult solution;
    candidate_statistics candidates;
    std::uint64_t map_hash{};
};
std::uint64_t publication_hash(std::span<const submap> maps);
operation_result<std::optional<optimization_result>> optimize_submaps(std::span<const submap> maps,
    const optimization_options& options, std::span<const absolute_measurement> measurements = {});
}
#endif
