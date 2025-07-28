#pragma once

#include <covoxslam/sensor_grid_cuda.hpp>
#include <covoxslam/sensor_grid_sampling.h>
#include <covoxslam/mapping.hpp>
#include <covoxslam/storage/schema.hpp>

namespace covoxslam::store
{
template<storage::layout, storage::schema> class coordinated_store;
}

namespace covoxslam::sensor_grid
{
enum class transfer_arithmetic : std::uint32_t { fp64, fp32 };
enum class source_mask_mode : std::uint32_t { dense, compact };
enum class discovery_mode : std::uint32_t { cpu, gpu_host, gpu_resident };
enum class candidate_mode : std::uint32_t { whole_tiles, voxel_mask, compacted_control, compact_fp32, append_fp32, compact_fp64, append_fp64, coarse_mask, whole_tile_control };
constexpr bool candidate_uses_mask(candidate_mode mode) noexcept
{
    return mode != candidate_mode::whole_tiles && mode != candidate_mode::whole_tile_control;
}
struct transfer_options
{
    source_mask_mode source_mask{source_mask_mode::dense};
    transfer_arithmetic arithmetic{transfer_arithmetic::fp64};
    bool specialized{};
    candidate_mode candidates{candidate_mode::whole_tiles};
    std::uint32_t candidate_capacity{1048576};
    discovery_mode discovery{discovery_mode::gpu_resident};
};

struct transfer_times
{
    double discovery_ms{}, resident_dispatch_ms{};
    float mask_ms{}, fusion_ms{}, candidates_ms{}, gpu_discovery_ms{}, gpu_allocation_ms{};
    std::uint64_t updates{}, candidate_voxels{}, kernel_launches{};
    std::uint64_t candidate_overflows{};
};
class gpu_map
{
    struct implementation;
    std::unique_ptr<implementation> state;
     gpu_map();
public:
    static operation_result<std::unique_ptr<gpu_map>> create(const lookup& table,
        std::uint32_t capacity = 32768, int device = 0, std::size_t maximum_bytes = 1073741824,
        transfer_options options = {}, bool resident_soa = true);
    ~gpu_map();
    gpu_map(const gpu_map&) = delete;
    gpu_map& operator=(const gpu_map&) = delete;
    operation_result<transfer_times> integrate(const gpu_grid& source, pose local_from_sensor,
        interpolation mode = interpolation::nearest);
    operation_result<tsdf_tiles> download();
    operation_result<std::size_t> retain(
        store::coordinated_store<storage::layout::pool_soa, storage::schema::esdf>& destination,
        std::uint64_t id, pose4 anchor);
    operation_status reset();
};
}
