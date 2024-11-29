#pragma once

#include <covoxslam/cuda_resident.hpp>
#include "ray_resident/types.h"
#include "ray_resident_reduce/types.h"
#include "ray_resident_step_tasks/types.h"
#include "ray_resident_warp/types.h"
#include "ray_resident_segmented/types.h"
#include <cuda_runtime_api.h>
#include <array>

namespace covoxslam
{
struct cuda_resident_context::implementation
{
    mapping_config config;
    cuda_resident_options options;
    detail::cuda_resident_view view;
    detail::cuda_resident_reduction_view reduction;
    detail::cuda_resident_task_view tasks;
    detail::cuda_segment_view segments;
    std::uint8_t* free_space_flags{};
    std::uint64_t next_scan{};
     detail::cuda_device_ray* rays{};
    cudaStream_t stream{};
    std::array<void*, 24> allocations{};
    std::size_t allocation_count{}, bytes{};
    bool failed{};
    struct staging_slot {
        detail::cuda_device_ray* rays{};
        cudaEvent_t complete{};
        std::array<cudaEvent_t, 4> task_events{};
        bool pending{};
    };
    std::array<staging_slot, 4> staging{};
    std::size_t next_slot{};
    cuda_resident_statistics statistics;

    operation_status check(int code)
    {
        if (code) {
            failed = true;
            return std::unexpected(operation_error{cudaGetErrorString(static_cast<cudaError_t>(code))});
        }
        return {};
    }

    operation_status integrate_reduction(std::size_t count, detail::cuda_ray_parameters parameters)
    {
        const auto mode = options.reduction;
        auto status = check(detail::cuda_begin_resident_reduction(view, reduction, mode, stream));
        const detail::cuda_free_space_view skip{free_space_flags, options.free_space.minimum_weight,
            !options.free_space.enabled || next_scan % options.free_space.refresh_scans == 0};

        if (options.free_space.enabled)
        {
            if (options.segment_length) {
                if (status) status = check(detail::cuda_walk_resident_segments(view, reduction, rays, count,
                    parameters, skip, segments, options.segment_length, stream));
            }
            else if (status) status = check(detail::cuda_walk_resident_skip(view, reduction, rays, count, parameters, skip, stream));
        }
        else {
        if (status) status = check(detail::cuda_walk_resident_reduction(view, reduction, rays, count, parameters, mode, 0, stream));
        if (options.warp.mode != cuda_resident_warp::none) {
            const detail::cuda_warp_parameters warp{options.warp.count,
                options.warp.mode == cuda_resident_warp::cache ? options.warp.cache_entries : 0, options.warp.scattered};

            if (status) status = check(detail::cuda_walk_resident_warp(view, reduction, rays, count, parameters, warp, stream));
        }
        else if (status) status = check(detail::cuda_walk_resident_reduction(view, reduction, rays, count, parameters, mode, 1, stream));
        }
        if (status && mode != cuda_resident_reduction::direct_atomic) {
            if (mode == cuda_resident_reduction::index)
                status = check(detail::cuda_scan_counts(reduction.ray_counts, reduction.ray_offsets, view.block_sums, count, stream));
            else status = check(detail::cuda_scan_counts(view.counts, view.offsets, view.block_sums,
                static_cast<std::size_t>(options.tile_capacity) * 512, stream));

            if (status) status = check(detail::cuda_walk_resident_reduction(view, reduction, rays, count, parameters, mode, 2, stream));
        }
        if (status) status = check(detail::cuda_finish_resident_reduction(view, reduction, mode, stream,
            {config.truncation_distance, config.maximum_weight, config.extraction_distance}, skip));
        return status;
    }

    operation_status harvest(staging_slot& slot)
    {
        if (!slot.pending) return {};

        if (options.step_tasks) {
            std::array<float, 3> elapsed{};

            for (std::size_t index = 0; index < elapsed.size(); ++index) {
                const auto status = check(cudaEventElapsedTime(&elapsed[index], slot.task_events[index], slot.task_events[index + 1]));
                if (!status) return status;
            }
            statistics.mapping.cuda_task_expansion_ms += elapsed[0];
             statistics.mapping.cuda_task_evaluation_ms += elapsed[1];
            statistics.mapping.cuda_task_fusion_ms += elapsed[2];
        }
        slot.pending = false;
        return {};
    }

    operation_status integrate_tasks(std::size_t count, detail::cuda_ray_parameters parameters, staging_slot& slot)
    {
        constexpr auto mode = cuda_resident_reduction::direct_atomic;
        auto status = check(cudaEventRecord(slot.task_events[0], stream));
        if (status) status = check(detail::cuda_begin_resident_reduction(view, reduction, mode, stream));

        if (status) status = check(detail::cuda_expand_resident_tasks(view, tasks, rays, count, parameters, stream));
        if (status) status = check(cudaEventRecord(slot.task_events[1], stream));

        if (status) status = check(detail::cuda_process_resident_tasks(view, reduction, tasks, rays, count, parameters, 0, stream));
        if (status) status = check(cudaEventRecord(slot.task_events[2], stream));
        if (status) status = check(detail::cuda_process_resident_tasks(view, reduction, tasks, rays, count, parameters, 1, stream));

        if (status) status = check(detail::cuda_finish_resident_reduction(view, reduction, mode, stream,
            {config.truncation_distance, config.maximum_weight, config.extraction_distance}));
        if (status) status = check(cudaEventRecord(slot.task_events[3], stream));
        return status;
    }

    template<class value_type>
    operation_status allocate(value_type*& pointer, std::size_t count)
    {

        if (allocation_count == allocations.size())
            return std::unexpected(operation_error{"resident allocation table is full"});
        auto& allocation = allocations[allocation_count];
        const auto result = check(cudaMalloc(&allocation, count * sizeof(value_type)));

        if (!result) return result;
        pointer = static_cast<value_type*>(allocation);
         ++allocation_count;
        return {};
    }

    ~implementation()
    {
        cudaSetDevice(options.device);
        if (stream) cudaStreamSynchronize(stream);

        for (auto& slot : staging) {

            for (auto event : slot.task_events)
                if (event) cudaEventDestroy(event);

            if (slot.complete) cudaEventDestroy(slot.complete);
            if (slot.rays) cudaFreeHost(slot.rays);
        }

        for (auto allocation : allocations)

            if (allocation) cudaFree(allocation);
        if (stream) cudaStreamDestroy(stream);
    }
};
}
