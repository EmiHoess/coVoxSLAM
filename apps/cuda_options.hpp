#pragma once

#include "cli_lookup.hpp"

namespace covoxslam
{
enum class cuda_cli_option { unknown, help, device, scan_count };
inline constexpr cli_option_entry<cuda_cli_option> cuda_cli_options[]{
    {"--help", cuda_cli_option::help},
     {"--device", cuda_cli_option::device},
    {"--scan-count", cuda_cli_option::scan_count}
};
}
