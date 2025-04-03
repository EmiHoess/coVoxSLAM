#pragma once
#include "layout_workload.hpp"
#include <covoxslam/storage/nested_array.hpp>

namespace covoxslam::metrics
{
struct nested_options
{
    measurement_options sampling;
    nested_order order{nested_order::lane_major};
    bool lane_schedule{}, cuda{};
};
#if defined(covoxslam_has_cuda)
template<schema s, std::size_t lanes, nested_array_policy policy, bool tagged>
operation_result<measurement> measure_nested_gpu(std::span<const aos_tile<s>> source, const nested_options& options);
#endif
template<schema s, std::size_t lanes, nested_array_policy policy, bool tagged>
operation_result<measurement> measure_nested(std::span<const aos_tile<s>> source, const nested_options& options)
{
#if defined(covoxslam_has_cuda)
    if (options.cuda) return measure_nested_gpu<s, lanes, policy, tagged>(source, options);
#endif
    const auto setup = clock::now();
    nested_draft<s, lanes, policy, tagged> owner;
    auto checked = owner.initialize(source.size(), options.order);
    if (!checked) return std::unexpected(checked.error());

    for (std::size_t t = 0; t < source.size(); ++t)
        for (std::size_t v = 0; v < 512; ++v) scatter_record(owner.view()[t].voxels[v], source[t].voxels[v]);
    std::vector<float> output(source.size() * 512), reference(output.size());
    field_query<s>(source, reference);
    measurement result;
    result.setup_ms = elapsed(setup);
    result.bytes = owner.allocated_bytes();
    const auto sample = [&]() -> operation_result<double> {
        const auto begin = clock::now();
        const auto query = [&](std::size_t tile, std::size_t voxel) {
            const auto flags = owner.template scalar<field_count<s> - 1>(tile, voxel);
            const auto distance = owner.template scalar<0>(tile, voxel);
            if constexpr (s == schema::tsdf)
                output[tile * 512 + voxel] = (flags & 1U) ? distance * owner.template scalar<1>(tile, voxel) : 0;
            else output[tile * 512 + voxel] = (flags & 1U) ? distance : 0;
        };
        if (options.lane_schedule) {
            for (std::size_t group = 0; group < owner.capacity() / lanes; ++group)
                for (std::size_t voxel = 0; voxel < 512; ++voxel)
                    for (std::size_t lane = 0; lane < lanes; ++lane)
                        if (group * lanes + lane < owner.size()) query(group * lanes + lane, voxel);
        }
        else {
            for (std::size_t tile = 0; tile < owner.size(); ++tile)
                for (std::size_t voxel = 0; voxel < 512; ++voxel) query(tile, voxel);
        }
        const auto duration = elapsed(begin);
        result.output_hash = hash_values<float>(output);
        return duration;
    };
    auto warmup = collect_warmup(options.sampling.stability, options.sampling.warmup, sample);
    if (!warmup) return std::unexpected(warmup.error());
    result.stability = std::move(*warmup);

    for (int i = 0; i < options.sampling.repeats; ++i) {
        auto value = sample();
        if (!value) return std::unexpected(value.error());
        result.samples_ms.push_back(*value);
    }

    for (std::size_t i = 0; i < output.size(); ++i)
        result.mismatch_count += std::memcmp(&output[i], &reference[i], 4) != 0;
    return result;
}
}
#if defined(covoxslam_has_cuda)
#include "nested_gpu.hpp"
#endif
