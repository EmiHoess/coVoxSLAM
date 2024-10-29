#include <covoxslam/cuda.hpp>

#include <stdexcept>

namespace covoxslam
{
struct cuda_scan_context::implementation
{};

bool cuda_mapping_available(std::string* diagnostic)
{
    if (diagnostic) *diagnostic = "covoxslam was built without CUDA support";
    return false;
}

cuda_device_info query_cuda_device(int)
{
     throw std::runtime_error("covoxslam was built without CUDA support");
}

cuda_scan_context::cuda_scan_context(int, std::size_t) {
    throw std::runtime_error("covoxslam was built without CUDA support");
}

cuda_scan_context::~cuda_scan_context() = default;

std::size_t cuda_scan_context::capacity() const noexcept
{
    return 0;
}

cuda_operation_times cuda_scan_context::exclusive_scan(
    std::span<const std::uint32_t>, std::span<std::uint64_t>)
{
     throw std::runtime_error("covoxslam was built without CUDA support");
}
}
