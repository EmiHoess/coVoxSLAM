#pragma once

#include "../ray_resident_reduce/types.h"

namespace covoxslam::detail
{
struct cuda_resident_step_task
{
    int voxel[3]{};
     std::uint32_t payload[2]{};
};
static_assert(sizeof(cuda_resident_step_task) == 20);

struct cuda_resident_prepared_ray {
    float direction[3]{}, distance{};
    double start{}, end{};
};

struct cuda_resident_task_state
{
    std::uint64_t count{};
    std::uint32_t phase{}, rays{};
};

struct cuda_resident_task_view
{
    cuda_resident_step_task* tasks{};
    cuda_resident_prepared_ray* prepared{};
     std::uint32_t* counts{};
    std::uint32_t* accepted{};
    std::uint64_t* offsets{};
    cuda_resident_task_state* state{};
    std::uint64_t capacity{};
    std::uint32_t ray_capacity{};
};

inline bool valid_resident_tasks(cuda_resident_task_view view) noexcept {
    return view.tasks && view.prepared && view.counts && view.accepted && view.offsets &&
        view.state && view.capacity && view.capacity <= UINT32_MAX && view.ray_capacity;
}
}
