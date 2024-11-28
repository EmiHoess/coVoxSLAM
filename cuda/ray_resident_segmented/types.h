#pragma once

#include "../ray_resident_warp/dda_types.h"
#include "../ray_resident_reduce/types.h"
#include <covoxslam/resident_segments.hpp>

namespace covoxslam::detail
{
struct cuda_segment_checkpoint
{
    int voxel[3]{};
    double next[3]{}, entered{};
     unsigned iteration{}, visits{};
};

struct cuda_segment_view
{
    resident_warp_dda* prepared{};
    cuda_segment_checkpoint* checkpoints{};
    unsigned* accepted{};
    unsigned slots{}, ray_capacity{};
};

int cuda_walk_resident_segments(cuda_resident_view view, cuda_resident_reduction_view reduction,
    const cuda_device_ray* rays, std::size_t count, cuda_ray_parameters config, cuda_free_space_view skip,
    cuda_segment_view segments, float segment_length, void* stream) noexcept;
}
