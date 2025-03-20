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
    if constexpr (l == layout::pooled_aos) checked = direct ? device.upload_direct(source) : device.upload(host);
    else checked = device.upload(host);
    if (!checked) return std::unexpected(checked.error());
    result.upload_ms = elapsed(upload_start);
    auto descriptor = device.descriptor();
    if (!descriptor) return std::unexpected(descriptor.error());
    const auto launch = [&] {
        return cuda_storage_status(static_cast<cudaError_t>(launch_layout_query(*descriptor, copy, resources.output, resources.stream)));
    };

    for (int i = 0; i < options.warmup; ++i) {
        checked = launch();
        if (!checked) return std::unexpected(checked.error());
    }
    checked = device.complete();
    if (!checked) return std::unexpected(checked.error());
    result.samples_ms.reserve(options.repeats);

    for (int i = 0; i < options.repeats; ++i)
    {
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
        result.samples_ms.push_back(milliseconds);
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
