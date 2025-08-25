#pragma once

#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/sensor/dense_field.hpp>
#include <covoxslam/sensor/lookup.hpp>
#include <covoxslam/sensor_grid_sampling.h>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/schema.hpp>
#include <covoxslam/voxel/sparse_map.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

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
static_assert(sizeof(float) == 4);
static_assert(sizeof(double) == 8);
static_assert(std::is_same_v<std::underlying_type_t<transfer_arithmetic>, std::uint32_t>);
static_assert(static_cast<std::uint32_t>(transfer_arithmetic::fp64) == 0);
static_assert(static_cast<std::uint32_t>(transfer_arithmetic::fp32) == 1);
static_assert(std::is_same_v<std::underlying_type_t<source_mask_mode>, std::uint32_t>);
static_assert(static_cast<std::uint32_t>(source_mask_mode::dense) == 0);
static_assert(static_cast<std::uint32_t>(source_mask_mode::compact) == 1);
static_assert(std::is_same_v<std::underlying_type_t<discovery_mode>, std::uint32_t>);
static_assert(static_cast<std::uint32_t>(discovery_mode::cpu) == 0);
static_assert(static_cast<std::uint32_t>(discovery_mode::gpu_host) == 1);
static_assert(static_cast<std::uint32_t>(discovery_mode::gpu_resident) == 2);
static_assert(std::is_same_v<std::underlying_type_t<candidate_mode>, std::uint32_t>);
static_assert(static_cast<std::uint32_t>(candidate_mode::whole_tiles) == 0);
static_assert(static_cast<std::uint32_t>(candidate_mode::voxel_mask) == 1);
static_assert(static_cast<std::uint32_t>(candidate_mode::compacted_control) == 2);
static_assert(static_cast<std::uint32_t>(candidate_mode::compact_fp32) == 3);
static_assert(static_cast<std::uint32_t>(candidate_mode::append_fp32) == 4);
static_assert(static_cast<std::uint32_t>(candidate_mode::compact_fp64) == 5);
static_assert(static_cast<std::uint32_t>(candidate_mode::append_fp64) == 6);
static_assert(static_cast<std::uint32_t>(candidate_mode::coarse_mask) == 7);
static_assert(static_cast<std::uint32_t>(candidate_mode::whole_tile_control) == 8);

static_assert(std::is_aggregate_v<transfer_options>);
static_assert(std::is_standard_layout_v<transfer_options>);
static_assert(std::is_trivially_copyable_v<transfer_options>);
static_assert(sizeof(transfer_options) == 24);
static_assert(alignof(transfer_options) == alignof(std::uint32_t));
static_assert(offsetof(transfer_options, source_mask) == 0);
static_assert(offsetof(transfer_options, arithmetic) == 4);
static_assert(offsetof(transfer_options, specialized) == 8);
static_assert(offsetof(transfer_options, candidates) == 12);
static_assert(offsetof(transfer_options, candidate_capacity) == 16);
static_assert(offsetof(transfer_options, discovery) == 20);
static_assert(transfer_options{}.source_mask == source_mask_mode::dense);
static_assert(transfer_options{}.arithmetic == transfer_arithmetic::fp64);
static_assert(!transfer_options{}.specialized);
static_assert(transfer_options{}.candidates == candidate_mode::whole_tiles);
static_assert(transfer_options{}.candidate_capacity == 1048576);
static_assert(transfer_options{}.discovery == discovery_mode::gpu_resident);

static_assert(std::is_aggregate_v<transfer_times>);
static_assert(std::is_standard_layout_v<transfer_times>);
static_assert(std::is_trivially_copyable_v<transfer_times>);
static_assert(sizeof(transfer_times) == 72);
static_assert(alignof(transfer_times) == alignof(double));
static_assert(offsetof(transfer_times, discovery_ms) == 0);
static_assert(offsetof(transfer_times, resident_dispatch_ms) == 8);
static_assert(offsetof(transfer_times, mask_ms) == 16);
static_assert(offsetof(transfer_times, gpu_allocation_ms) == 32);
static_assert(offsetof(transfer_times, updates) == 40);
static_assert(offsetof(transfer_times, candidate_voxels) == 48);
static_assert(offsetof(transfer_times, kernel_launches) == 56);
static_assert(offsetof(transfer_times, candidate_overflows) == 64);
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
static_assert(sizeof(gpu_map) == sizeof(void*));
static_assert(alignof(gpu_map) == alignof(void*));
static_assert(!std::is_copy_constructible_v<gpu_map>);
static_assert(!std::is_copy_assignable_v<gpu_map>);
static_assert(!std::is_move_constructible_v<gpu_map>);
static_assert(!std::is_move_assignable_v<gpu_map>);
}
