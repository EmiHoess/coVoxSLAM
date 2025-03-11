#pragma once
#include <covoxslam/pgo_snapshot.hpp>

#if defined(covoxslam_has_pgo)
namespace covoxslam::pgo
{
enum class publication_mode { apply, shadow };
struct publication_limits
{
    double maximum_translation_m{20.0};
     double maximum_yaw_rad{1.5707963267948966};
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
