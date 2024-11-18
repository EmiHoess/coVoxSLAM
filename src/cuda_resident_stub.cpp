#include <covoxslam/cuda_resident.hpp>

namespace covoxslam
{
struct cuda_resident_context::implementation {};
cuda_resident_context::cuda_resident_context() = default;
cuda_resident_context::~cuda_resident_context() = default;
std::size_t cuda_resident_context::device_bytes() const noexcept { return 0; }
cuda_resident_order cuda_resident_context::fusion_order() const noexcept { return cuda_resident_order::ordered; }
operation_result<std::unique_ptr<cuda_resident_context>> cuda_resident_context::create(
    mapping_config, cuda_resident_options)
{
    return std::unexpected(operation_error{"resident mapping requires a CUDA enabled build"});
}
operation_status cuda_resident_context::submit(std::span<const ray>)
{
    return std::unexpected(operation_error{"resident mapping requires a CUDA enabled build"});
}
operation_result<cuda_resident_statistics> cuda_resident_context::flush()
{
    return std::unexpected(operation_error{"resident mapping requires a CUDA enabled build"});
}
operation_result<tsdf_tiles> cuda_resident_context::download()
{
    return std::unexpected(operation_error{"resident mapping requires a CUDA enabled build"});
}
operation_status cuda_resident_context::reset()
{
    return std::unexpected(operation_error{"resident mapping requires a CUDA enabled build"});
}
operation_status cuda_resident_context::restore(const tsdf_tiles&)
{
    return std::unexpected(operation_error{"resident mapping requires a CUDA enabled build"});
}
}
