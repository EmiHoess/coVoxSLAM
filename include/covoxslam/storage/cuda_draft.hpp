#pragma once
#include <covoxslam/storage/device_buffer.hpp>

namespace covoxslam::storage
{
template<schema s, layout l> using cuda_draft = device_buffer<s, l>;
}
