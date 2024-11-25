#pragma once

#include <covoxslam/cuda_resident.hpp>
#include "cli_lookup.hpp"
#include <charconv>
#include <chrono>
#include <iostream>

namespace covoxslam
{
enum class resident_option { unknown, backend, tiles, rays, contributions, warp_count, warp_cache, warp_order };
inline constexpr cli_option_entry<resident_option> resident_options[]{
    {"--backend", resident_option::backend}, {"--resident-tiles", resident_option::tiles},
     {"--resident-rays", resident_option::rays}, {"--resident-contributions", resident_option::contributions},
    {"--warp-count", resident_option::warp_count}, {"--warp-cache-entries", resident_option::warp_cache},
    {"--warp-ray-order", resident_option::warp_order}
};
static_assert(valid_cli_options(resident_options));
enum class resident_backend { unknown, ordered, arrival, reduce_index, reduce_segment, reduce_atomic, step_tasks, warp_match, warp_cache };
inline constexpr cli_option_entry<resident_backend> resident_backends[]{
    {"cuda-ray-resident", resident_backend::ordered},
     {"cuda-ray-resident-unordered", resident_backend::arrival},
    {"cuda-ray-resident-unordered-reduce-index", resident_backend::reduce_index},
    {"cuda-ray-resident-unordered-reduce-segment", resident_backend::reduce_segment},
    {"cuda-ray-resident-unordered-reduce-atomic", resident_backend::reduce_atomic},
    {"cuda-ray-resident-step-tasks", resident_backend::step_tasks},
    {"cuda-ray-resident-warp-match", resident_backend::warp_match}, {"cuda-ray-resident-warp-cache", resident_backend::warp_cache}
};
static_assert(valid_cli_options(resident_backends));
enum class warp_ray_order { unknown, input, scattered };
inline constexpr cli_option_entry<warp_ray_order> warp_ray_orders[]{
    {"input", warp_ray_order::input}, {"scattered", warp_ray_order::scattered}
};
static_assert(valid_cli_options(warp_ray_orders));
struct resident_cli_options
{
    cuda_resident_options mapper;
    bool selected{}, specified{}, warp_specified{};
};

inline operation_result<bool> parse_resident_option(std::string_view name, std::string_view value,
    resident_cli_options& options)
{
    const auto id = lookup_cli_option(name, resident_options);

    if (id == resident_option::unknown) return false;
    if (id == resident_option::backend) {
        options.mapper.reduction = cuda_resident_reduction::none;
        options.mapper.step_tasks = false;
        options.mapper.warp.mode = cuda_resident_warp::none;
        options.mapper.order = cuda_resident_order::arrival;
        switch (lookup_cli_option(value, resident_backends))
        {
        case resident_backend::ordered: options.mapper.order = cuda_resident_order::ordered; break;
        case resident_backend::arrival: options.mapper.order = cuda_resident_order::arrival; break;
        case resident_backend::reduce_index: options.mapper.reduction = cuda_resident_reduction::index; break;
        case resident_backend::reduce_segment: options.mapper.reduction = cuda_resident_reduction::segment; break;
        case resident_backend::reduce_atomic: options.mapper.reduction = cuda_resident_reduction::direct_atomic; break;
        case resident_backend::warp_match:
        case resident_backend::warp_cache:
            options.mapper.reduction = cuda_resident_reduction::direct_atomic;
            options.mapper.warp.mode = lookup_cli_option(value, resident_backends) == resident_backend::warp_match ?
                cuda_resident_warp::match : cuda_resident_warp::cache;
            break;
        case resident_backend::step_tasks:
            options.mapper.reduction = cuda_resident_reduction::direct_atomic;
             options.mapper.step_tasks = true;
            break;
        case resident_backend::unknown: options.selected = false; return false;
        }
        options.selected = true;
        return true;
    }
    if (id == resident_option::warp_order) {
        switch (lookup_cli_option(value, warp_ray_orders)) {
        case warp_ray_order::input: options.mapper.warp.scattered = false; break;
         case warp_ray_order::scattered: options.mapper.warp.scattered = true; break;
        default: return std::unexpected(operation_error{"expected input or scattered warp ray order"});
        }
        options.warp_specified = true;
        return true;
    }
    std::uint32_t count{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);

    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || (!count && id != resident_option::contributions))
        return std::unexpected(operation_error{"resident capacities require uint32 values and positive tile and ray limits"});
    switch (id)
    {
    case resident_option::tiles: options.mapper.tile_capacity = count; break;
    case resident_option::rays: options.mapper.ray_capacity = count; break;
    case resident_option::contributions: options.mapper.contribution_capacity = count; break;
    case resident_option::warp_count:
        if (count > 65536) return std::unexpected(operation_error{"warp count exceeds 65536"});
        options.mapper.warp.count = count;
        options.warp_specified = true;
        return true;
    case resident_option::warp_cache:
        if (count != 32 && count != 64 && count != 128 && count != 256)
            return std::unexpected(operation_error{"expected 32, 64, 128 or 256 warp cache entries"});
        options.mapper.warp.cache_entries = count;
        options.warp_specified = true;
        return true;
    default: return false;
    }
    options.specified = true;
    return true;
}

inline pipeline_callbacks resident_callbacks(cuda_resident_context& mapper, mapping_config config,
    std::optional<operation_error>& failure, std::function<void(const submap&)> publish,
    std::function<void(submap&)> finalize)
{
    pipeline_callbacks callbacks;
    callbacks.cancelled = [&failure] { return failure.has_value(); };
    callbacks.begin = [&mapper, &failure](const submap_frame&)
    {
        const auto status = mapper.reset();

        if (!status) failure = status.error();
    };
    callbacks.integrate_scan = [&mapper, &failure](const submap_frame&, const frame_descriptor&,
        std::span<const ray> rays, const pose&) {
        const auto status = mapper.submit(rays);
        if (!status) failure = status.error();
    };

    callbacks.finish = [&mapper, &failure, config, publish, finalize](const submap_frame& frame)
    {
        auto tiles = mapper.download();

        if (!tiles) { failure = tiles.error(); return; }
        const auto statistics = mapper.flush();
        if (!statistics) { failure = statistics.error(); return; }
        submap completed;
        completed.frame = frame;
        completed.optimized_world_pose = frame.anchor;
         completed.config = config;
        completed.tsdf = std::move(*tiles);
        completed.statistics = statistics->mapping;

        if (finalize) finalize(completed);
        else if (config.esdf_generator == esdf_backend::cpu_blocks_batch)
        {
            const auto start = std::chrono::steady_clock::now();
            build_cpu_esdf(completed.tsdf, completed.esdf, config);
            completed.statistics.esdf_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
        }

        if (publish) publish(completed);
        std::cout << "tsdf " << frame.id << ": " << completed.tsdf.size() << " tiles, "
                  << statistics->mapping.voxel_updates << " updates, " << statistics->batches
                   << " resident batches, " << statistics->explicit_waits << " explicit waits\n";
        std::cout << resident_fusion_name(mapper.fusion_order(), mapper.reduction_mode(), mapper.uses_step_tasks(), mapper.warp_configuration().mode) << ", "
                  << statistics->mapping.cuda_voxel_segments << " voxel groups, " << statistics->maximum_segment_length
                  << " maximum contributions per voxel, " << mapper.device_bytes() << " device bytes\n";
        if (mapper.uses_step_tasks())
            std::cout << "task expansion " << statistics->mapping.cuda_task_expansion_ms << " ms, evaluation "
                      << statistics->mapping.cuda_task_evaluation_ms << " ms, fusion "
                      << statistics->mapping.cuda_task_fusion_ms << " ms\n";
        if (mapper.warp_configuration().mode != cuda_resident_warp::none)
            std::cout << statistics->warp_flush_groups << " warp flush groups, " << statistics->warp_peer_groups
                      << " peer groups, " << statistics->warp_cache_hits << " cache hits, "
                      << statistics->warp_cache_flushes << " cache flushes\n";
        std::cout << "esdf " << frame.id << ": " << esdf_backend_name(config.esdf_generator)
                  << ", " << completed.esdf.size() << " tiles, " << completed.statistics.esdf_ms << " ms\n";
    };
    return callbacks;
}
}
