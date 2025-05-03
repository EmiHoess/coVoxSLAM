#pragma once
#include <covoxslam/storage/schema.hpp>
#include <covoxslam/result.hpp>

namespace covoxslam::store
{
enum class mapping_method { cpu_ray, ordered_cuda, combined_nearest };
inline operation_status check_mapping_support(mapping_method method, storage::layout layout,
    storage::schema field, bool initialized_topology)
{
    if (unsigned(layout) > unsigned(storage::layout::pool_soa) || field != storage::schema::tsdf)
        return std::unexpected(operation_error{"mapping layout or field is unsupported"});
#if !defined(covoxslam_has_urd)
    if (layout != storage::layout::pooled_aos)
        return std::unexpected(operation_error{"mapping layout requires external URD"});
#endif
    switch (method) {
    case mapping_method::cpu_ray: return {};
    case mapping_method::ordered_cuda:
    case mapping_method::combined_nearest:
#if defined(covoxslam_has_cuda)
        if (!initialized_topology) return std::unexpected(operation_error{"GPU mapping requires initialized destination keys"});
        return {};
#else
        return std::unexpected(operation_error{"GPU mapping requires CUDA"});
#endif
    default: return std::unexpected(operation_error{"unknown mapping method"});
    }
}
}
