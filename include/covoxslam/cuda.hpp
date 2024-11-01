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

namespace covoxslam
{
struct ray;
namespace detail { struct cuda_ray_device_view; struct cuda_ray_contribution; }

struct cuda_ray_config
{
    float voxel_size{0.20F};
    float truncation_distance{0.60F};
     float minimum_ray_length_m{0.10F};
    float maximum_ray_length_m{16.0F};
    bool voxel_carving{true};
    bool allow_clear{};
};

struct cuda_contribution_statistics {
    std::uint64_t submitted_rays{};
    std::uint64_t contributing_rays{};
    std::uint64_t contributions{};
     std::uint64_t uploaded_bytes{};
    std::uint64_t downloaded_bytes{};
    float upload_ms{};
    float count_ms{};
    float scan_ms{};
    float emit_ms{};
};

class cuda_ray_context
{
    struct implementation;
    std::unique_ptr<implementation> state;
public:
    cuda_ray_context(int device, std::size_t ray_capacity,
                     std::uint64_t contribution_capacity, cuda_ray_config config = {});
     ~cuda_ray_context();
    cuda_ray_context(const cuda_ray_context&) = delete;
    cuda_ray_context& operator=(const cuda_ray_context&) = delete;
    std::size_t ray_capacity() const noexcept;
    std::uint64_t contribution_capacity() const noexcept;
    cuda_contribution_statistics generate(std::span<const ray> rays);
    detail::cuda_ray_device_view device_view() const noexcept;
    void copy_contributions(std::span<detail::cuda_ray_contribution> output);
}
;
}

#include <covoxslam/mapping.hpp>

namespace covoxslam
{
struct cuda_mapper_options {
    int device{};
    std::size_t ray_capacity{65536};
     std::size_t contribution_capacity{8388608};
    std::size_t maximum_device_bytes{2ULL << 30};
};

class cuda_mapper
{
    struct implementation;
    std::unique_ptr<implementation> state;
public:
    explicit cuda_mapper(mapping_config config = {}, cuda_mapper_options options = {});
     ~cuda_mapper();
    cuda_mapper(const cuda_mapper&) = delete;
    cuda_mapper& operator=(const cuda_mapper&) = delete;
    void begin(const submap_frame& frame);
    void integrate_scan(std::span<const ray> rays);
    submap finish(const submap_frame& frame);
    bool is_active() const noexcept;
}
;
}
