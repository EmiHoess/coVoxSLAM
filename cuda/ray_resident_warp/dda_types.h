#pragma once

#include "../ray_resident/types.h"

namespace covoxslam::detail
{
struct resident_warp_dda
{
    cuda_device_ray ray{};
    float direction[3]{}, distance{};
    int voxel[3]{}, step[3]{};
     double next[3]{}, delta[3]{}, start{}, end{}, entered{};
    unsigned iteration{}, maximum_steps{}, contributions{};
    bool active{};
};
}
