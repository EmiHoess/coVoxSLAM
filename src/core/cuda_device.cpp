#include <covoxslam/core/cuda_device.hpp>
#include <cuda_runtime_api.h>

namespace covoxslam
{
namespace
{
operation_status cuda_status(cudaError_t error)
{
    if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
    return {};
}
}
operation_result<cuda_device_info> query_cuda_device_checked(int index)
{
    int count{};
    if (const auto status = cuda_status(cudaGetDeviceCount(&count)); !status) return std::unexpected(status.error());
    if (index < 0 || index >= count) return std::unexpected(operation_error{"CUDA device index is unavailable"});
    cudaDeviceProp properties{};
    if (const auto status = cuda_status(cudaGetDeviceProperties(&properties, index)); !status) return std::unexpected(status.error());
    cuda_device_info result{index, properties.major, properties.minor};
    result.name = properties.name;
    result.total_memory_bytes = properties.totalGlobalMem;
    if (const auto status = cuda_status(cudaRuntimeGetVersion(&result.runtime_version)); !status) return std::unexpected(status.error());
    if (const auto status = cuda_status(cudaDriverGetVersion(&result.driver_version)); !status) return std::unexpected(status.error());
    return result;
}

bool cuda_mapping_available(std::string* diagnostic)
{
    const auto result = query_cuda_device_checked(0);
    if (diagnostic) *diagnostic = result ? std::string{} : result.error().message;
    return result.has_value();
}
}
