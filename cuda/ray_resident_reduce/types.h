#pragma once

#include "../ray_resident/types.h"
#include <covoxslam/resident_policy.hpp>

namespace covoxslam::detail
{
struct cuda_resident_reduction_sample { float distance{}, weight{}; };
static_assert(sizeof(cuda_resident_reduction_sample) == 8);

struct cuda_resident_reduction_view
{
    cuda_resident_reduction_sample* samples{};
    std::uint32_t* voxel_indices{};
     std::uint32_t* ray_counts{};
    std::uint64_t* ray_offsets{};
    double* weights{};
    double* weighted_distances{};
    std::uint32_t ray_capacity{};
};

inline bool valid_reduction_storage(cuda_resident_reduction_view view,
    cuda_resident_reduction mode) noexcept
{
    switch (mode)
    {
    case cuda_resident_reduction::index:
        return view.samples && view.voxel_indices && view.ray_counts && view.ray_offsets &&
             view.ray_capacity && view.weights && view.weighted_distances;
    case cuda_resident_reduction::segment: return view.samples != nullptr;
    case cuda_resident_reduction::direct_atomic: return view.weights && view.weighted_distances;
    default: return false;
    }
}

int cuda_begin_resident_reduction(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_resident_reduction mode, void* stream) noexcept;
int cuda_finish_resident_reduction(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_resident_reduction mode, void* stream, cuda_fusion_parameters config) noexcept;
int cuda_walk_resident_reduction(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters parameters,
    cuda_resident_reduction mode, unsigned stage, void* stream) noexcept;

inline bool valid_resident_reduction_view(cuda_resident_view view, cuda_resident_reduction_view reduction,
    cuda_resident_reduction mode) noexcept
{
    return valid_resident_map(view) && valid_reduction_storage(reduction, mode) &&
        (mode == cuda_resident_reduction::direct_atomic || (view.contribution_capacity && view.block_sums &&
        (mode != cuda_resident_reduction::segment || view.offsets)));
}

inline cuda_resident_reduction_view resident_reduction_storage(cuda_resident_reduction_view view,
    cuda_resident_reduction mode) noexcept
{
    if (mode != cuda_resident_reduction::index) {
        view.ray_counts = nullptr;
         view.ray_capacity = 0;
    }

    if (mode == cuda_resident_reduction::segment) view.weights = view.weighted_distances = nullptr;
    return view;
}
}
