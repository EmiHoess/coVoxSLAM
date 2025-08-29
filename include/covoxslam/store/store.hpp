#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/map_store/authority.hpp>
#include <covoxslam/map_store/coordinator.hpp>
#include <covoxslam/map_store/cpu_index.hpp>
#include <covoxslam/map_store/residency.hpp>
#include <covoxslam/store/field_access.hpp>
#include <covoxslam/storage/host_buffer.hpp>
#include <covoxslam/storage/layout.hpp>
#include <covoxslam/storage/record_pack.hpp>
#include <cmath>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace covoxslam::store
{
template<storage::layout l, storage::schema es = storage::schema::esdf> class map_store
{
#include <covoxslam/map_store/detail/cpu_store_state.inl>
#include <covoxslam/map_store/detail/cpu_store_snapshot.inl>
#include <covoxslam/map_store/detail/cpu_store_fields.inl>
};
}
