#pragma once
#include <covoxslam/storage/device_collection.hpp>

namespace covoxslam::storage
{
struct device_descriptor
{
    std::byte* fields[5]{};
    std::size_t size_tiles{}, capacity_tiles{}, tile_stride{}, voxel_stride{};
    std::uint64_t generation{};
    schema record_schema{schema::tsdf};
    layout storage_layout{layout::pool_soa};
    int device{-1};
};
template<schema s, bool constant = false> struct device_accessor
{
    device_descriptor descriptor;
    static constexpr bool read_only = constant;
    covoxslam_storage_hd std::size_t size() const noexcept { return descriptor.size_tiles; }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return descriptor.capacity_tiles; }
    covoxslam_storage_hd bool valid() const noexcept
    {
        return descriptor.record_schema == s && descriptor.storage_layout == layout::pool_soa &&
            descriptor.tile_stride == 2048 && descriptor.voxel_stride == 4 &&
            descriptor.generation != 0 && size() <= capacity();
    }
#if defined(__CUDACC__)
    template<std::size_t field>
    __device__ scalar_reference<constant, field_type<s, field>> scalar(std::size_t tile, std::size_t voxel) const noexcept
    {
        using scalar_type = std::conditional_t<constant, const field_type<s, field>, field_type<s, field>>;
        return *reinterpret_cast<scalar_type*>(descriptor.fields[field] + tile * descriptor.tile_stride + voxel * descriptor.voxel_stride);
    }
#endif
};
template<schema s, bool constant = false> using device_map_view = device_collection<s, device_accessor<s, constant>>;
int initialize_device_fields(device_descriptor descriptor, void* stream) noexcept;
}
