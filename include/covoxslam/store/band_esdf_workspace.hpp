#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/store/band_esdf.hpp>
#include <covoxslam/store/esdf_resources.hpp>

namespace covoxslam::store
{
inline operation_result<std::size_t> band_workspace_bytes(std::size_t count, std::size_t maximum)
{
    constexpr std::size_t per_tile = 512 * 20 + sizeof(block_key) +
        sizeof(detail::cuda_tile_coordinate) + 2 * sizeof(unsigned);

    if (!count || count > UINT32_MAX / 512 || maximum < 8 || count > (maximum - 8) / per_tile)
        return std::unexpected(operation_error{"band ESDF workspace exceeds its limit"});
    return count * per_tile + 8;
}
struct band_esdf_workspace
{
    std::shared_ptr<budget_claim> field_memory;
    storage::cuda_draft<storage::schema::projective_esdf, storage::layout::pool_soa> fields;
    esdf_scratch metadata;
     std::size_t reserved_bytes{};
    band_esdf_workspace() = default;
    band_esdf_workspace(const band_esdf_workspace&) = delete;
    band_esdf_workspace& operator=(const band_esdf_workspace&) = delete;
    operation_status complete()
    {
        return storage::cuda_storage_status(cudaStreamSynchronize(metadata.stream));
    }
    operation_status initialize(std::size_t count, std::size_t maximum,
        memory_budget& budget, cudaStream_t stream)
    {
        if (field_memory) return std::unexpected(operation_error{"band workspace is already allocated"});
        auto total = band_workspace_bytes(count, maximum);
        if (!total) return std::unexpected(total.error());
        const auto bytes = count * 512 * 20;
        auto claimed = budget.reserve({0, 0, bytes});

        if (!claimed) return std::unexpected(claimed.error());
        field_memory = std::move(*claimed);
        auto checked = fields.initialize(count);

        if (!checked) return checked;
        const std::array<std::size_t, 4> sizes{count * sizeof(block_key),
            count * sizeof(detail::cuda_tile_coordinate), count * 2 * sizeof(unsigned), 8};
        checked = metadata.initialize(sizes, maximum - bytes, budget, stream);
        if (!checked) return checked;
        reserved_bytes = *total;
        return {};
    }
};
}
