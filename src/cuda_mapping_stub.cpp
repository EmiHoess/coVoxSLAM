#include <covoxslam/cuda.hpp>

namespace covoxslam
{
bool cuda_mapping_available(std::string* diagnostic)
{
    if (diagnostic) *diagnostic = "covoxslam was built without CUDA support";
    return false;
}

operation_result<cuda_device_info> query_cuda_device_checked(int)
{
    return std::unexpected(operation_error{"covoxslam was built without CUDA support"});
}
}
