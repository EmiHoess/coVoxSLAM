#pragma once
#include <covoxslam/storage/device_view.hpp>

namespace covoxslam::metrics
{
int launch_layout_query(storage::device_descriptor descriptor, bool copy, void* output, void* stream) noexcept;
}
