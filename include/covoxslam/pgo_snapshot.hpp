#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/optimization/registration.hpp>
#include <covoxslam/optimization/solver.hpp>

#include <cstdint>
#include <optional>
#include <span>

#if defined(covoxslam_has_pgo)
namespace covoxslam::pgo
{
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
