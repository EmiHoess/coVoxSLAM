#pragma once
#include "layout_workload.hpp"

namespace covoxslam::metrics
{
template<schema s, bool copy, class view_type>
operation_result<measurement> measure_cpu(view_type view, std::span<const aos_tile<s>> source,
    const measurement_options& options)
{
    if (options.warmup < 0 || options.repeats <= 0 || options.repeats > 1000 || options.warmup > 1000)
        return std::unexpected(operation_error{"invalid layout measurement counts"});
    using output_type = std::conditional_t<copy, value<s>, float>;
    std::vector<output_type> output(source.size() * 512), reference(output.size());
    const auto run = [&] {
        if constexpr (copy) record_copy<s>(view, output);
        else field_query<s>(view, output);
    };
    if constexpr (copy) record_copy<s>(source, reference);
    else field_query<s>(source, reference);
    measurement result;
    result.bytes = output.size() * sizeof(output_type);
    result.samples_ms.reserve(options.repeats);

    for (int i = 0; i < options.warmup; ++i) run();

    for (int i = 0; i < options.repeats; ++i) {
        const auto start = clock::now();
        run();
        result.samples_ms.push_back(elapsed(start));
        result.output_hash = hash_values<output_type>(output);
    }

    for (std::size_t i = 0; i < output.size(); ++i)
        result.mismatch_count += std::memcmp(&output[i], &reference[i], sizeof(output_type)) != 0;
    return result;
}
template<schema s, layout l> operation_status fill_draft(draft<s, l>& owner, std::span<const aos_tile<s>> source)
{
    auto initialized = owner.initialize(source.size());
    if (!initialized) return initialized;
    auto view = owner.view();

    for (std::size_t tile = 0; tile < source.size(); ++tile)
        for (std::size_t voxel = 0; voxel < voxels_per_tile; ++voxel)
            scatter_record(view[tile].voxels[voxel], source[tile].voxels[voxel]);
    return {};
}
template<schema s, layout l, bool copy> operation_result<measurement> measure_host_layout(
    std::span<const aos_tile<s>> source, const measurement_options& options)
{
    const auto start = clock::now();
    draft<s, l> owner;
    auto filled = fill_draft(owner, source);
    if (!filled) return std::unexpected(filled.error());
    const auto setup = elapsed(start);
    auto measured = measure_cpu<s, copy>(std::as_const(owner).view(), source, options);
    if (measured) measured->setup_ms = setup;
    return measured;
}
}
