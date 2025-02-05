#include "sensor_grid_map_internal.hpp"
#include <algorithm>

namespace covoxslam::sensor_grid
{
operation_result<std::vector<int3>> gpu_map::implementation::discover_gpu(field_view source,
    detail::cuda_projective_pose transform, float& elapsed)
{
    auto status = check(cudaEventRecord(events[6], stream));
    if (status) status = check(cuda_sensor_discover(source, mask, transform, discovery, stream));
    if (status) status = check(cudaEventRecord(events[7], stream));
    discovery_state result{};
    std::vector<std::uint64_t> staging(discovery.capacity);
    if (status) status = check(cudaMemcpyAsync(&result, discovery.state, sizeof(result), cudaMemcpyDeviceToHost, stream));
    if (status) status = check(cudaMemcpyAsync(staging.data(), discovery.keys,
        staging.size() * 8, cudaMemcpyDeviceToHost, stream));
    status = wait(status);
    if (status) status = check(cudaEventElapsedTime(&elapsed, events[6], events[7]));

    if (!status) return std::unexpected(status.error());
    if (result.error || result.count > discovery.capacity)
        return std::unexpected(operation_error{"sensor GPU discovery exceeds coordinate or tile capacity"});
    staging.resize(result.count);
    std::sort(staging.begin(), staging.end());
    std::vector<int3> output;
    output.reserve(staging.size());

    for (std::size_t i = 0; i < staging.size(); ++i) {

        if (staging[i] == UINT64_MAX || (i && staging[i - 1] == staging[i]))
            return std::unexpected(operation_error{"invalid sensor GPU discovery ownership"});
        const auto key = decode_discovery_key(staging[i]);
        output.push_back({key.x, key.y, key.z});
    }
    return output;
}
}
