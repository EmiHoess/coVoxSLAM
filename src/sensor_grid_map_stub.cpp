#include <covoxslam/sensor_grid_map.hpp>
namespace covoxslam::sensor_grid
{
struct gpu_map::implementation {};
gpu_map::gpu_map() = default;
gpu_map::~gpu_map() = default;
operation_result<std::unique_ptr<gpu_map>> gpu_map::create(const lookup&, std::uint32_t, int, std::size_t, transfer_options, bool)
{
    return std::unexpected(operation_error{"sensor TSDF requires CUDA support"});
}
operation_result<transfer_times> gpu_map::integrate(const gpu_grid&, pose, interpolation)
{
    return std::unexpected(operation_error{"sensor TSDF requires CUDA support"});
}
operation_result<tsdf_tiles> gpu_map::download() {
    return std::unexpected(operation_error{"sensor TSDF requires CUDA support"});
}
operation_status gpu_map::reset()
{
    return std::unexpected(operation_error{"sensor TSDF requires CUDA support"});
}
operation_result<std::size_t> gpu_map::retain(
    store::coordinated_store<storage::layout::pool_soa, storage::schema::esdf>&,
    std::uint64_t, pose4)
{
    return std::unexpected(operation_error{"resident retention requires CUDA support"});
}
}
