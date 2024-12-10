#include <covoxslam/sensor_grid_cuda.hpp>

namespace covoxslam::sensor_grid
{
struct gpu_grid::implementation {};
gpu_grid::gpu_grid() = default;
gpu_grid::~gpu_grid() = default;
operation_result<std::unique_ptr<gpu_grid>> gpu_grid::create(const lookup&, int, std::size_t)
{
    return std::unexpected(operation_error{"sensor field requires CUDA support"});
}
std::size_t gpu_grid::device_bytes() const noexcept { return 0; }
field_view gpu_grid::device_field() const noexcept { return {}; }
}
