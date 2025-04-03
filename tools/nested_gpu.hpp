#pragma once
#include "layout_gpu.hpp"
#include "nested_kernel.hpp"
#include <covoxslam/storage/cuda_nested.hpp>

namespace covoxslam::metrics
{
template<schema s, std::size_t lanes, nested_array_policy policy, bool tagged>
operation_result<measurement> measure_nested_gpu(std::span<const aos_tile<s>> source, const nested_options& options)
{
    const auto setup = clock::now();
    nested_draft<s, lanes, policy, tagged> host;
    auto checked = host.initialize(source.size(), options.order);
    if (!checked) return std::unexpected(checked.error());

    for (std::size_t t = 0; t < source.size(); ++t)
        for (std::size_t v = 0; v < 512; ++v) scatter_record(host.view()[t].voxels[v], source[t].voxels[v]);
    cuda_nested_draft<s, lanes, policy, tagged> device;
    checked = device.initialize(host);
    if (!checked) return std::unexpected(checked.error());
    std::vector<float> output(source.size() * 512), reference(output.size());
    field_query<s>(source, reference);
    query_resources resources;
    resources.stream = device.stream();
    checked = cuda_storage_status(cudaMalloc(&resources.output, output.size() * sizeof(float)));
    if (!checked) return std::unexpected(checked.error());
    checked = cuda_storage_status(cudaEventCreate(&resources.begin));
    if (!checked) return std::unexpected(checked.error());
    checked = cuda_storage_status(cudaEventCreate(&resources.end));
    if (!checked) return std::unexpected(checked.error());
    measurement result;
    result.bytes = host.allocated_bytes();
    result.setup_ms = elapsed(setup);
    const auto upload = clock::now();
    checked = device.upload(host);
    if (!checked) return std::unexpected(checked.error());
    result.upload_ms = elapsed(upload);
    auto descriptor = device.descriptor();
    if (!descriptor) return std::unexpected(descriptor.error());
    const auto sample = [&]() -> operation_result<double> {
        auto status = cudaEventRecord(resources.begin, resources.stream);
        if (status == cudaSuccess) status = static_cast<cudaError_t>(launch_nested_query(*descriptor, options.lane_schedule,
            static_cast<float*>(resources.output), resources.stream));
        if (status == cudaSuccess) status = cudaEventRecord(resources.end, resources.stream);
        if (status == cudaSuccess) status = cudaEventSynchronize(resources.end);
        float milliseconds{};
        if (status == cudaSuccess) status = cudaEventElapsedTime(&milliseconds, resources.begin, resources.end);
        if (status != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(status)});
        return double(milliseconds);
    };
    auto warmup = collect_warmup(options.sampling.stability, options.sampling.warmup, sample);
    if (!warmup) return std::unexpected(warmup.error());
    result.stability = std::move(*warmup);

    for (int i = 0; i < options.sampling.repeats; ++i) {
        auto measured = sample();
        if (!measured) return std::unexpected(measured.error());
        result.samples_ms.push_back(*measured);
        const auto download = clock::now();
        checked = cuda_storage_status(cudaMemcpyAsync(output.data(), resources.output, output.size() * sizeof(float), cudaMemcpyDeviceToHost, resources.stream));
        if (!checked) return std::unexpected(checked.error());
        checked = cuda_storage_status(cudaStreamSynchronize(resources.stream));
        if (!checked) return std::unexpected(checked.error());
        result.download_ms += elapsed(download);
        result.output_hash = hash_values<float>(output);

        for (std::size_t j = 0; j < output.size(); ++j)
            result.mismatch_count += std::memcmp(&output[j], &reference[j], 4) != 0;
    }
    result.download_ms /= options.sampling.repeats;
    return result;
}
}
