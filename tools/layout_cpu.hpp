#pragma once
#include "layout_workload.hpp"
#include <covoxslam/storage/ranges.hpp>
#include <covoxslam/storage/record_pack.hpp>
#include <covoxslam/storage/field_streams.hpp>

namespace covoxslam::metrics
{
template<class function> operation_status select_packing_batch(const measurement_options& options, function&& action)
{
    if (!options.fixed_batch) return action(packing_options{options.batch_records});
    switch (options.batch_records) {
    case 1: return action(fixed_packing<1>());
    case 8: return action(fixed_packing<8>());
    case 16: return action(fixed_packing<16>());
    case 32: return action(fixed_packing<32>());
    case 64: return action(fixed_packing<64>());
    case 128: return action(fixed_packing<128>());
    default: return std::unexpected(operation_error{"fixed packing batch must be 1, 8, 16, 32, 64 or 128"});
    }
}
template<schema s, bool copy, class view_type>
operation_result<measurement> measure_cpu(view_type view, std::span<const aos_tile<s>> source,
    const measurement_options& options)
{
    if (options.warmup < 0 || options.repeats <= 0 || options.repeats > 1000 || options.warmup > 1000)
        return std::unexpected(operation_error{"invalid layout measurement counts"});
    using output_type = std::conditional_t<copy, value<s>, float>;
    const auto count = source.size() * 512;
    std::vector<std::size_t> indices;
    if (!copy && options.query == query_method::indexed) {
        if (!options.query_stride) return std::unexpected(operation_error{"query stride must be positive"});

        for (std::size_t i = 0; i < count; i += options.query_stride) indices.push_back(i);
    }
    std::vector<output_type> output(!copy && options.query == query_method::indexed ? indices.size() : count), reference(output.size());
    const auto run = [&]() -> operation_status {
        if constexpr (copy) {
            if (options.method != copy_method::scalar) {
                if constexpr (requires { view.native_accessor(); }) {
                    if (options.method == copy_method::packed)
                        return select_packing_batch(options, [&](auto packing) {
                            return pack_records(view, 0, output.size(), std::span<value<s>>(output), packing);
                        });
                    return copy_records(view, 0, output.size(), std::span<value<s>>(output));
                }
                else {
                    for (std::size_t tile = 0; tile < view.size(); ++tile)
                        std::memcpy(output.data() + tile * 512, view[tile].voxels.data(), sizeof(aos_tile<s>));
                }
            }
            else record_copy<s>(view, output);
        }
        else {
            if constexpr (requires { view.native_accessor(); }) {
                switch (options.query) {
                case query_method::tile: return selective_query(view, {0, count}, output, stream_order::tile);
                case query_method::contiguous: return selective_query(view, {0, count}, output, stream_order::contiguous);
                case query_method::indexed: return gather_field_indices<s, 0>(view, indices, output);
                default: break;
                }
            }
            if (options.query == query_method::indexed) {
                for (std::size_t i = 0; i < indices.size(); ++i)
                    output[i] = std::get<0>(scalar_fields<s>(view[indices[i] / 512].voxels[indices[i] % 512]));
            }
            else field_query<s>(view, output);
        }
        return {};
    };
    if constexpr (copy) record_copy<s>(source, reference);
    else {
        if (options.query == query_method::indexed) {
            for (std::size_t i = 0; i < indices.size(); ++i)
                reference[i] = std::get<0>(scalar_fields<s>(source[indices[i] / 512].voxels[indices[i] % 512]));
        }
        else field_query<s>(source, reference);
    }
    measurement result;
    result.bytes = output.size() * sizeof(output_type);
    result.samples_ms.reserve(options.repeats);

    for (int i = 0; i < options.warmup; ++i) {
        auto checked = run();
        if (!checked) return std::unexpected(checked.error());
    }

    for (int i = 0; i < options.repeats; ++i) {
        const auto start = clock::now();
        auto checked = run();
        result.samples_ms.push_back(elapsed(start));
        if (!checked) return std::unexpected(checked.error());
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
    if constexpr (copy) {
        if (options.method == copy_method::unpacked) {
            std::vector<value<s>> input(source.size() * 512), output(input.size());
            record_copy<s>(source, input);
            measurement result;
            result.setup_ms = elapsed(start);
            result.bytes = input.size() * sizeof(value<s>);
            result.samples_ms.reserve(options.repeats);
            const auto run = [&] {
                return select_packing_batch(options, [&](auto packing) {
                    return unpack_records(std::span<const value<s>>(input), owner.view(), 0, packing);
                });
            };

            for (int i = 0; i < options.warmup + options.repeats; ++i)
            {
                const auto begin = clock::now();
                auto checked = run();
                const auto milliseconds = elapsed(begin);
                if (!checked) return std::unexpected(checked.error());
                record_copy<s>(std::as_const(owner).view(), output);
                result.output_hash = hash_values<value<s>>(output);
                if (i >= options.warmup) result.samples_ms.push_back(milliseconds);
            }

            for (std::size_t i = 0; i < input.size(); ++i)
                result.mismatch_count += std::memcmp(&input[i], &output[i], sizeof(value<s>)) != 0;
            return result;
        }
    }
    auto measured = measure_cpu<s, copy>(std::as_const(owner).view(), source, options);
    if (measured) measured->setup_ms = setup;
    return measured;
}
}
