#pragma once

#include <cstddef>
#include <cstdint>
#include <covoxslam/core/result.hpp>
#include <string>

namespace covoxslam
{
struct cuda_device_info
{
    int index{};
    int compute_major{};
     int compute_minor{};
    int runtime_version{};
    int driver_version{};
    std::size_t total_memory_bytes{};
    std::string name;
};

bool cuda_mapping_available(std::string* diagnostic = nullptr);
operation_result<cuda_device_info> query_cuda_device_checked(int index);
}
