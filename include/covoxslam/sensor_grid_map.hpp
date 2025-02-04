#pragma once

#include <covoxslam/sensor_grid_cuda.hpp>
#include <covoxslam/sensor_grid_sampling.h>
#include <covoxslam/sensor_grid_masks.hpp>
#include <covoxslam/mapping.hpp>

namespace covoxslam::sensor_grid
{
struct transfer_times
{
    double discovery_ms{};
    float mask_ms{}, fusion_ms{}, candidates_ms{}, gpu_discovery_ms{};
    std::uint64_t updates{}, candidate_voxels{}, kernel_launches{};
    std::uint64_t candidate_overflows{};
};
class gpu_map
{
    struct implementation;
    std::unique_ptr<implementation> state;
     gpu_map();
public:
    static operation_result<std::unique_ptr<gpu_map>> create(const lookup& table,
        std::uint32_t capacity = 32768, int device = 0, std::size_t maximum_bytes = 1073741824,
        transfer_options options = {});
    ~gpu_map();
    gpu_map(const gpu_map&) = delete;
    gpu_map& operator=(const gpu_map&) = delete;
    operation_result<transfer_times> integrate(const gpu_grid& source, pose local_from_sensor,
        interpolation mode = interpolation::trilinear);
    operation_result<tsdf_tiles> download();
    operation_status reset();
};
operation_result<std::vector<int3>> destination_tiles(geometry grid, float size,
    std::span<const std::uint8_t> observed, pose local_from_sensor, std::size_t maximum_tiles);
}
