#include <covoxslam/sensor_grid_cuda.hpp>

namespace covoxslam::sensor_grid
{
struct gpu_grid::implementation {};
gpu_grid::gpu_grid() = default;
gpu_grid::~gpu_grid() = default;
operation_result<std::unique_ptr<gpu_grid>> gpu_grid::create(const lookup&, int, std::size_t, gather_options)
{
    return std::unexpected(operation_error{"sensor field requires CUDA support"});
}
std::size_t gpu_grid::device_bytes() const noexcept { return 0; }
int gpu_grid::device_index() const noexcept { return -1; }
gather_statistics gpu_grid::statistics() const noexcept { return {}; }
field_view gpu_grid::device_field() const noexcept { return {}; }
operation_result<update_times> gpu_grid::update(std::span<const float>, gather_policy)
{
    return std::unexpected(operation_error{"sensor field requires CUDA support"});
}
operation_result<std::vector<voxel>> gpu_grid::download()
{
    return std::unexpected(operation_error{"sensor field requires CUDA support"});
}
}
