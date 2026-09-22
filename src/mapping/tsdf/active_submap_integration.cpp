#include "active_submap_internal.hpp"
#include <chrono>
#include <cmath>

namespace covoxslam::sensor_grid
{
operation_result<transfer_times> gpu_map::integrate(const gpu_grid& source, pose transform, interpolation mode)
{
    auto& data = *state;
    const auto field = source.device_field();
    const auto q = transform.rotation;

    if (data.failed || !field.voxels || source.device_index() != data.device || mode != interpolation::nearest ||
        !data.soa_selected || data.options.discovery != discovery_mode::gpu_resident ||
        data.options.source_mask != source_mask_mode::dense || data.options.candidates != candidate_mode::whole_tiles ||
        data.options.arithmetic != transfer_arithmetic::fp64 || data.options.specialized ||
        field.side != data.grid.side || field.origin != data.grid.origin || field.voxel_size != data.settings.voxel_size ||
        !is_unit_quaternion(q) || !std::isfinite(transform.translation.x) ||
        !std::isfinite(transform.translation.y) || !std::isfinite(transform.translation.z))
        return std::unexpected(operation_error{"invalid sensor TSDF transfer"});
    transform.rotation = normalized(q);
    const auto rotation = transform.rotation;
    const detail::cuda_projective_pose packed{{transform.translation.x, transform.translation.y, transform.translation.z},
        {rotation.x, rotation.y, rotation.z, rotation.w}};
    auto status = data.check(cudaSetDevice(data.device));
    if (status) status = data.check(cudaEventRecord(data.mask_start, data.stream));
    if (status) status = data.check(cuda_sensor_mask(field, data.mask, data.stream));
    if (status) status = data.check(cudaEventRecord(data.mask_end, data.stream));
    if (status) return data.integrate_resident(field, packed, mode);
    return std::unexpected(status.error());
}
}
