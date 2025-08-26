#pragma once
#include <covoxslam/storage/cuda_draft.hpp>

namespace covoxslam::storage
{
template<schema s, layout l> using device_buffer = cuda_draft<s, l>;
}
