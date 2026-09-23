#pragma once
#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/pgo/registration.hpp>
#include <covoxslam/pgo/solver.hpp>

#include <cstdint>
#include <numbers>
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
enum class publication_mode { apply, shadow };
struct publication_limits
{
    double maximum_translation_m{20.0};
     double maximum_yaw_rad{std::numbers::pi_v<double> / 2.0};
    double maximum_registration_growth{0.1};
    double minimum_registration_retention{0.5};
};
class publication_update
{
    std::uint64_t source_hash_{};
    std::vector<pose4> poses_;
    publication_update(std::uint64_t source_hash, std::vector<pose4> poses)
        : source_hash_(source_hash), poses_(std::move(poses)) {}
    friend operation_result<publication_update> prepare_publication(std::span<const submap>,
        const optimization_result&, const publication_limits&);
    friend operation_status publish_poses(std::span<submap>, const publication_update&, publication_mode);
public:
    std::span<const pose4> poses() const noexcept { return poses_; }
};
operation_result<publication_update> prepare_publication(std::span<const submap> maps,
    const optimization_result& result, const publication_limits& limits = {});
operation_status publish_poses(std::span<submap> maps, const publication_update& update,
    publication_mode mode = publication_mode::apply);
}
#endif
