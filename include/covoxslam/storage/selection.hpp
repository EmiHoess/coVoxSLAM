#pragma once
#include <covoxslam/storage/draft.hpp>
#include <string_view>

namespace covoxslam::storage
{
constexpr bool layout_available(layout selected) noexcept
{
    switch (selected) {
    case layout::pooled_aos: return true;
#if defined(covoxslam_has_urd)
    case layout::tile_aosoa:
    case layout::pool_soa: return true;
#endif
    default: return false;
    }
}
constexpr std::string_view layout_name(layout selected) noexcept
{
    switch (selected)
    {
    case layout::pooled_aos: return "pooled-aos";
    case layout::tile_aosoa: return "tile-aosoa";
    case layout::pool_soa: return "pool-soa";
    default: return "invalid";
    }
}
inline operation_result<layout> parse_layout(std::string_view name)
{
    for (auto selected : {layout::pooled_aos, layout::tile_aosoa, layout::pool_soa})
    {
        if (name != layout_name(selected)) continue;
        if (!layout_available(selected))
            return std::unexpected(operation_error{"layout requires optional URD support"});
        return selected;
    }
    return std::unexpected(operation_error{"unknown storage layout"});
}
template<class function> operation_status dispatch_layout(layout selected, function&& action)
{
    switch (selected)
    {
    case layout::pooled_aos: return action.template operator()<layout::pooled_aos>();
#if defined(covoxslam_has_urd)
    case layout::tile_aosoa: return action.template operator()<layout::tile_aosoa>();
    case layout::pool_soa: return action.template operator()<layout::pool_soa>();
#endif
    default: return std::unexpected(operation_error{"storage layout is unavailable"});
    }
}
}
