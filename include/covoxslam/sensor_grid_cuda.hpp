#pragma once

#include <covoxslam/sensor_grid.hpp>
#include <memory>

namespace covoxslam::sensor_grid
{
struct update_times { float upload_ms{}, gather_ms{}; };
class gpu_grid
{
    struct implementation;
    std::unique_ptr<implementation> state;
    gpu_grid();
public:
    static operation_result<std::unique_ptr<gpu_grid>> create(const lookup& table,
        int device = 0, std::size_t maximum_bytes = 1073741824);
    ~gpu_grid();
     gpu_grid(const gpu_grid&) = delete;
    gpu_grid& operator=(const gpu_grid&) = delete;
    std::size_t device_bytes() const noexcept;
    field_view device_field() const noexcept;
    operation_result<update_times> update(std::span<const float> ranges, gather_policy policy = {});
    operation_result<std::vector<voxel>> download();
};
}
