#pragma once
#include <covoxslam/storage/host_buffer.hpp>
#include <covoxslam/storage/views.hpp>

namespace covoxslam::storage
{
template<schema s, layout l> using draft = host_buffer<s, l>;
}
