#include <covoxslam/cuda.hpp>

#include <stdexcept>

namespace covoxslam
{
struct cuda_scan_context::implementation
{};

bool cuda_mapping_available(std::string* diagnostic)
{
    if (diagnostic) *diagnostic = "covoxslam was built without CUDA support";
    return false;
}

cuda_device_info query_cuda_device(int)
{
     throw std::runtime_error("covoxslam was built without CUDA support");
}

cuda_scan_context::cuda_scan_context(int, std::size_t) {
    throw std::runtime_error("covoxslam was built without CUDA support");
}

cuda_scan_context::~cuda_scan_context() = default;

std::size_t cuda_scan_context::capacity() const noexcept
{
    return 0;
}

cuda_operation_times cuda_scan_context::exclusive_scan(
    std::span<const std::uint32_t>, std::span<std::uint64_t>)
{
     throw std::runtime_error("covoxslam was built without CUDA support");
}
}

#include "sparse_mapping.h"

namespace covoxslam
{
struct cuda_ray_context::implementation
{};

cuda_ray_context::cuda_ray_context(int, std::size_t, std::uint64_t, cuda_ray_config)
{
    throw std::runtime_error("covoxslam was built without CUDA support");
}

cuda_ray_context::~cuda_ray_context() = default;
std::size_t cuda_ray_context::ray_capacity() const noexcept { return 0; }
std::uint64_t cuda_ray_context::contribution_capacity() const noexcept { return 0; }
detail::cuda_ray_device_view cuda_ray_context::device_view() const noexcept { return {}; }

void cuda_ray_context::copy_contributions(std::span<detail::cuda_ray_contribution>)
{
    throw std::runtime_error("covoxslam was built without CUDA support");
}
}

#include <covoxslam/mapping.hpp>

namespace covoxslam
{
cuda_contribution_statistics cuda_ray_context::generate(std::span<const ray>)
{
    throw std::runtime_error("covoxslam was built without CUDA support");
}
}

namespace covoxslam
{
struct cuda_mapper::implementation
{};

cuda_mapper::cuda_mapper(mapping_config, cuda_mapper_options)
{
    throw std::runtime_error("covoxslam was built without CUDA support");
}

cuda_mapper::~cuda_mapper() = default;
bool cuda_mapper::is_active() const noexcept { return false; }
void cuda_mapper::begin(const submap_frame&)
{
    throw std::runtime_error("covoxslam was built without CUDA support");
}
}

namespace covoxslam
{
void cuda_mapper::integrate_scan(std::span<const ray>)
{
    throw std::runtime_error("covoxslam was built without CUDA support");
}

submap cuda_mapper::finish(const submap_frame&)
{
    throw std::runtime_error("covoxslam was built without CUDA support");
}
}

namespace covoxslam
{
cuda_site_esdf_statistics build_cuda_site_esdf(const tsdf_tiles&, esdf_tiles&,
    const mapping_config&, const cuda_site_esdf_options&)
{
    throw std::runtime_error("covoxslam was built without CUDA support");
}
}

namespace covoxslam
{
cuda_voxblox_statistics build_cuda_voxblox_esdf(const tsdf_tiles&, esdf_tiles&,
    const mapping_config&, const cuda_voxblox_options&)
{
    throw std::runtime_error("CUDA Voxblox ESDF is unavailable in this build");
}
}

namespace covoxslam
{
struct cuda_projective_mapper::implementation {};
cuda_projective_mapper::cuda_projective_mapper(const sensor&, mapping_config, cuda_projective_options)
{
    throw std::runtime_error("CUDA projective mapping is unavailable in this build");
}
cuda_projective_mapper::~cuda_projective_mapper() = default;
}

namespace covoxslam
{
void cuda_projective_mapper::begin(const submap_frame&)
{
    throw std::runtime_error("CUDA projective mapping is unavailable in this build");
}
void cuda_projective_mapper::integrate_scan(std::span<const ray>, const pose&)
{
    throw std::runtime_error("CUDA projective mapping is unavailable in this build");
}
bool cuda_projective_mapper::is_active() const noexcept { return false; }
}
