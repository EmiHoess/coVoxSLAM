#pragma once
#include <covoxslam/storage/views.hpp>

namespace covoxslam::storage
{
struct nested_descriptor
{
    std::byte* fields[5]{};
    std::size_t size_tiles{}, capacity_tiles{}, lanes{}, group_stride{}, tile_stride{}, voxel_stride{}, member_stride{};
    std::uint64_t generation{};
    schema record_schema{schema::tsdf};
    int device{-1};
};
static_assert(std::is_trivially_copyable_v<nested_descriptor>);
template<schema s, bool constant = false> struct nested_device_accessor
{
    nested_descriptor descriptor;
    static constexpr bool read_only = constant;
    covoxslam_storage_hd std::size_t size() const noexcept { return descriptor.size_tiles; }
    covoxslam_storage_hd std::size_t capacity() const noexcept { return descriptor.capacity_tiles; }
    covoxslam_storage_hd bool valid() const noexcept
    {
        return descriptor.generation && descriptor.record_schema == s && descriptor.device >= 0 &&
            descriptor.lanes && descriptor.size_tiles <= descriptor.capacity_tiles && (!size() || descriptor.fields[0]);
    }
#if defined(__CUDACC__)
    template<std::size_t field> __device__ auto& scalar(std::size_t tile, std::size_t voxel) const noexcept
    {
        using scalar_type = std::conditional_t<constant, const field_type<s, field>, field_type<s, field>>;
        const auto offset = tile / descriptor.lanes * descriptor.group_stride +
            tile % descriptor.lanes * descriptor.tile_stride + voxel * descriptor.voxel_stride;
        return *reinterpret_cast<scalar_type*>(descriptor.fields[field] + offset);
    }
#endif
};
int initialize_nested_fields(nested_descriptor descriptor, void* stream);
}
