#pragma once

#include <covoxslam/cuda_resident.hpp>
#include "cli_lookup.hpp"
#include <charconv>
#include <chrono>
#include <iostream>

namespace covoxslam
{
enum class resident_option { unknown, backend, tiles, rays, contributions };
inline constexpr cli_option_entry<resident_option> resident_options[]{
    {"--backend", resident_option::backend}, {"--resident-tiles", resident_option::tiles},
     {"--resident-rays", resident_option::rays}, {"--resident-contributions", resident_option::contributions}
};
static_assert(valid_cli_options(resident_options));
enum class resident_backend { unknown, ordered, arrival };
inline constexpr cli_option_entry<resident_backend> resident_backends[]{
    {"cuda-ray-resident", resident_backend::ordered},
     {"cuda-ray-resident-unordered", resident_backend::arrival}
};
static_assert(valid_cli_options(resident_backends));
struct resident_cli_options
{
    cuda_resident_options mapper;
    bool selected{}, specified{};
};

inline operation_result<bool> parse_resident_option(std::string_view name, std::string_view value,
    resident_cli_options& options)
{
    const auto id = lookup_cli_option(name, resident_options);

    if (id == resident_option::unknown) return false;
    if (id == resident_option::backend) {
        switch (lookup_cli_option(value, resident_backends))
        {
        case resident_backend::ordered: options.mapper.order = cuda_resident_order::ordered; break;
        case resident_backend::arrival: options.mapper.order = cuda_resident_order::arrival; break;
        case resident_backend::unknown: options.selected = false; return false;
        }
        options.selected = true;
        return true;
    }
    std::uint32_t count{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), count);

    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !count)
        return std::unexpected(operation_error{"resident capacities must be positive uint32 values"});
    switch (id)
    {
    case resident_option::tiles: options.mapper.tile_capacity = count; break;
    case resident_option::rays: options.mapper.ray_capacity = count; break;
    case resident_option::contributions: options.mapper.contribution_capacity = count; break;
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
        std::cout << "esdf " << frame.id << ": " << esdf_backend_name(config.esdf_generator)
                  << ", " << completed.esdf.size() << " tiles, " << completed.statistics.esdf_ms << " ms\n";
    };
    return callbacks;
}
}
