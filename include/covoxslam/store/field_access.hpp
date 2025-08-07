#pragma once
#include <covoxslam/storage/schema.hpp>

namespace covoxslam::store
{
template<storage::schema s, storage::schema es, class entry_type>
constexpr auto& select_field(entry_type& entry) noexcept
{

    if constexpr (s == storage::schema::tsdf)
        return entry.tsdf;
     else return entry.esdf;
}
}
