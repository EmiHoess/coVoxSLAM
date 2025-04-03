#pragma once
#include <covoxslam/storage/nested_device_view.hpp>
namespace covoxslam::metrics
{
int launch_nested_query(storage::nested_descriptor descriptor, bool lane_schedule, float* output, void* stream);
}
