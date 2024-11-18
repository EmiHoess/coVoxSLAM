#pragma once

#include <covoxslam/cuda.hpp>
#include <covoxslam/resident_policy.hpp>

namespace covoxslam
{
struct cuda_resident_options
{
    int device{};
    std::uint32_t tile_capacity{4096};
     std::uint32_t ray_capacity{65536};
    std::uint32_t contribution_capacity{8388608};
    std::size_t maximum_device_bytes{2ULL << 30};
    bool voxel_carving{true};
    bool allow_clear{};
    cuda_resident_order order{cuda_resident_order::ordered};
};

struct cuda_resident_statistics
{
    mapping_statistics mapping;
    std::uint64_t batches{}, explicit_waits{};
     std::uint32_t maximum_segment_length{};
};

class cuda_resident_context
{
    struct implementation;
    std::unique_ptr<implementation> state;
    cuda_resident_context();
public:
    static operation_result<std::unique_ptr<cuda_resident_context>> create(
        mapping_config config = {}, cuda_resident_options options = {});
    ~cuda_resident_context();
    cuda_resident_context(const cuda_resident_context&) = delete;
     cuda_resident_context& operator=(const cuda_resident_context&) = delete;
    std::size_t device_bytes() const noexcept;
    cuda_resident_order fusion_order() const noexcept;
    operation_status submit(std::span<const ray> rays);
    operation_result<cuda_resident_statistics> flush();
    operation_result<tsdf_tiles> download();
    operation_status reset();
    operation_status restore(const tsdf_tiles& tiles);
};
}
