#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
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

struct cuda_operation_times {
    float upload_ms{};
    float scan_ms{};
     float download_ms{};
};

bool cuda_mapping_available(std::string* diagnostic = nullptr);
cuda_device_info query_cuda_device(int index);

class cuda_scan_context
{
    struct implementation;
    std::unique_ptr<implementation> state;
public:
    cuda_scan_context(int device, std::size_t capacity);
     ~cuda_scan_context();
    cuda_scan_context(const cuda_scan_context&) = delete;
    cuda_scan_context& operator=(const cuda_scan_context&) = delete;
    std::size_t capacity() const noexcept;
    cuda_operation_times exclusive_scan(
        std::span<const std::uint32_t> counts, std::span<std::uint64_t> offsets);
}
;
}
