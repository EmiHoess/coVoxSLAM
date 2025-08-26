#pragma once
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/views.hpp>

namespace covoxslam::storage
{
template<schema s, class accessor> struct device_voxel_array
{
    accessor access;
    std::size_t tile{};
    covoxslam_storage_hd constexpr std::size_t size() const noexcept { return 512; }
#if defined(__CUDACC__)
    __device__ auto operator[](std::size_t voxel) const noexcept
    {
        if constexpr (s == schema::tsdf)
            return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
                access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel), access.template scalar<3>(tile, voxel)};
        else
            return voxel_reference<s, accessor::read_only>{access.template scalar<0>(tile, voxel),
                access.template scalar<1>(tile, voxel), access.template scalar<2>(tile, voxel),
                access.template scalar<3>(tile, voxel), access.template scalar<4>(tile, voxel)};
    }
#endif
};
template<schema s, class accessor> struct device_tile_view { device_voxel_array<s, accessor> voxels; };
template<schema s, class accessor> class device_collection
{
    accessor access_;
public:
    covoxslam_storage_hd explicit device_collection(accessor access) noexcept : access_(access) {}
    covoxslam_storage_hd std::size_t size() const noexcept { return access_.size(); }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return access_.capacity(); }
    covoxslam_storage_hd bool empty() const noexcept { return size() == 0; }
    covoxslam_storage_hd bool valid() const noexcept { return access_.valid(); }
    covoxslam_storage_hd const accessor& native_accessor() const noexcept { return access_; }
#if defined(__CUDACC__)
    __device__ auto operator[](std::size_t tile) const noexcept
    {
        return device_tile_view<s, accessor>{{access_, tile}};
    }
#endif
};
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
