#pragma once
#include <covoxslam/pgo/pose_graph.hpp>
#include <covoxslam/pipeline/submap.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#if defined(covoxslam_has_pgo)
#include <ceronoceres/ceronoceres.hpp>
namespace covoxslam::pgo
{
struct surface_bounds
{
    std::array<double, 3> minimum{}, maximum{};
    bool valid{};
};
struct surface_sample {
    ceronoceres::Vec3d point;
    double distance{}, weight{};
     std::uint64_t ordinal{};
};
struct directed_candidate
{
    std::size_t source{}, target{};
    std::uint64_t diagnostic_id{};
    std::vector<surface_sample> samples;
};
struct candidate_options
{
    std::size_t sample_stride{16};
    double overlap_padding_m{0.6};
};
struct candidate_set
{
    std::vector<surface_bounds> bounds;
    std::vector<directed_candidate> directions;
};
operation_result<candidate_set> find_candidates(std::span<const submap> maps, candidate_options options = {});
operation_status validate_candidates(const candidate_set& candidates, std::size_t map_count);
struct candidate_statistics
{
    std::size_t surface_maps{}, overlap_pairs{}, directed_groups{}, samples{};
};
operation_result<candidate_statistics> summarize_candidates(const candidate_set& candidates);
struct registration_options
{
    ceronoceres::Loss loss{ceronoceres::LossKind::Huber, 0.20};
    double minimum_weight{1.0e-6};
    double maximum_weight{10000.0};
    double minimum_gradient{1.0e-6};
    double maximum_residual_m{1.2};
    bool nearest_fallback{};
};
operation_status append_registration(graph_snapshot& graph, std::span<const submap> maps,
    const candidate_set& candidates, registration_options options = {}, bool copy_fields = true);
}
#endif
