#pragma once

#include <expected>
#include <string>

namespace covoxslam
{
struct operation_error
{
     std::string message;
};

template<class value_type>
using operation_result = std::expected<value_type, operation_error>;

using operation_status = operation_result<void>;
}
