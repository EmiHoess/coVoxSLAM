#pragma once
#include "layout_cpu.hpp"
#include "layout_kernel.hpp"
#include <covoxslam/storage/cuda_draft.hpp>

namespace covoxslam::metrics
{
struct query_resources
{
    void* output{};
    cudaStream_t stream{};
    cudaEvent_t begin{}, end{};
    ~query_resources()
    {
        if (stream) cudaStreamSynchronize(stream);
        if (output) cudaFree(output);
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }
};
template<schema s, layout l, bool copy> operation_result<measurement> measure_gpu(
    std::span<const aos_tile<s>> source, const measurement_options& options, bool direct = false)
{
    if (source.empty() || options.warmup < 0 || options.warmup > 1000 || options.repeats <= 0 || options.repeats > 1000)
        return std::unexpected(operation_error{"invalid GPU measurement extent or repetitions"});
    if (direct && l != layout::pooled_aos) return std::unexpected(operation_error{"direct input requires AoS"});
    const auto started = clock::now();
    draft<s, l> host;
    auto checked = fill_draft(host, source);
    if (!checked) return std::unexpected(checked.error());
    cuda_draft<s, l> device;
    checked = device.initialize(source.size());
    if (!checked) return std::unexpected(checked.error());
    using output_type = std::conditional_t<copy, value<s>, float>;
    std::vector<output_type> output(source.size() * 512), reference(output.size());
    if constexpr (copy) record_copy<s>(source, reference);
    else field_query<s>(source, reference);
    query_resources resources;
    resources.stream = device.stream();
    const auto bytes = output.size() * sizeof(output_type);
    checked = cuda_storage_status(cudaMalloc(&resources.output, bytes));
    if (!checked) return std::unexpected(checked.error());
    checked = cuda_storage_status(cudaEventCreate(&resources.begin));
    if (!checked) return std::unexpected(checked.error());
    checked = cuda_storage_status(cudaEventCreate(&resources.end));
    if (!checked) return std::unexpected(checked.error());
    measurement result;
    result.setup_ms = elapsed(started);
    result.bytes = bytes;
    result.copy_calls = direct ? 1 : source.size() * field_count<s>;
    const auto upload_start = clock::now();
    if (options.bulk_transfers && !direct) {
        transfer_measurement transfer;
        checked = device.upload_range(host, 0, 0, source.size() * 512, (1U << field_count<s>) - 1,
            transfer_method::bulk_same_layout, transfer);
        result.copy_calls = transfer.calls;
        result.transfer_bytes = transfer.useful_bytes;
        result.transfer_waits = transfer.waits;
        result.transfer_fallback = transfer.fallback;
        result.planning_ms = transfer.planning_ms;
        result.submission_ms = transfer.submission_ms;
        result.completion_ms = transfer.completion_ms;
    }
    else {
        if constexpr (l == layout::pooled_aos) checked = direct ? device.upload_direct(source) : device.upload(host);
        else checked = device.upload(host);
        result.planning_ms = result.submission_ms = result.completion_ms = -1;
        result.transfer_bytes = source.size() * sizeof(aos_tile<s>);
        result.transfer_waits = 1;
    }
    if (!checked) return std::unexpected(checked.error());
    result.upload_ms = elapsed(upload_start);
    auto descriptor = device.descriptor();
    if (!descriptor) return std::unexpected(descriptor.error());
    const auto launch = [&] {
        return cuda_storage_status(static_cast<cudaError_t>(launch_layout_query(*descriptor, copy, resources.output, resources.stream)));
    };

    const auto sample = [&]() -> operation_result<double> {
        checked = cuda_storage_status(cudaEventRecord(resources.begin, resources.stream));
        if (!checked) return std::unexpected(checked.error());
        checked = launch();
        if (!checked) return std::unexpected(checked.error());
        checked = cuda_storage_status(cudaEventRecord(resources.end, resources.stream));
        if (!checked) return std::unexpected(checked.error());
        checked = cuda_storage_status(cudaEventSynchronize(resources.end));
        if (!checked) return std::unexpected(checked.error());
        float milliseconds{};
        checked = cuda_storage_status(cudaEventElapsedTime(&milliseconds, resources.begin, resources.end));
        if (!checked) return std::unexpected(checked.error());
        return double(milliseconds);
    };
    auto warmup = collect_warmup(options.stability, options.warmup, sample);
    if (!warmup) return std::unexpected(warmup.error());
    result.stability = std::move(*warmup);
    result.samples_ms.reserve(options.repeats);

    for (int i = 0; i < options.repeats; ++i) {
        auto measured = sample();
        if (!measured) return std::unexpected(measured.error());
        result.samples_ms.push_back(*measured);
        const auto download_start = clock::now();
        checked = cuda_storage_status(cudaMemcpyAsync(output.data(), resources.output, bytes, cudaMemcpyDeviceToHost, resources.stream));
        if (!checked) return std::unexpected(checked.error());
        checked = device.complete();
        if (!checked) return std::unexpected(checked.error());
        result.download_ms += elapsed(download_start);
        result.output_hash = hash_values<output_type>(output);

        for (std::size_t j = 0; j < output.size(); ++j)
            result.mismatch_count += std::memcmp(&output[j], &reference[j], sizeof(output_type)) != 0;
    }
    result.download_ms /= options.repeats;
    return result;
}
inline const char* workload_name(bool copy, layout selected, bool direct)
{
    if (!copy) return "field_query";
    return direct || selected == layout::pooled_aos ? "aos_record_copy" : "layout_to_aos_conversion";
}
}
