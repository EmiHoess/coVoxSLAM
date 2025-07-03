#include <covoxslam/cuda.hpp>
#include "cuda_options.hpp"

#include <charconv>
#include <iostream>
#include <limits>
#include <string_view>
int main(int argc, char** argv)
{
    const auto report_error = [](std::string_view message) {
        std::cerr << "covoxslam_cuda: " << message << '\n';
        return 1;
    };
        int device{};
         for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            const auto option_id = covoxslam::lookup_cli_option(option, covoxslam::cuda_cli_options);
            switch (option_id)
            {
            case covoxslam::cuda_cli_option::help:
                std::cout << "Usage: covoxslam_cuda [--device N]\n";
                return 0;
            case covoxslam::cuda_cli_option::unknown:
                return report_error("expected --device or --scan-count and its value");
            case covoxslam::cuda_cli_option::device:
                break;
            case covoxslam::cuda_cli_option::scan_count:
                if (index + 1 >= argc) return report_error("missing --scan-count value");
                return report_error("--scan-count is not applicable to coVoxSLAM");
            }
            if (++index >= argc)
                return report_error("expected --device or --scan-count and its value");
            const std::string_view text(argv[index]);
            std::size_t value{};
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
             if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
                return report_error("expected an unsigned decimal integer");
            switch (option_id) {
            case covoxslam::cuda_cli_option::device:

                if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    return report_error("device index exceeds its integer range");
                device = static_cast<int>(value);
                break;
            default: return report_error("expected --device or --scan-count and its value");
            }
        }
        const auto queried = covoxslam::query_cuda_device_checked(device);
        if (!queried) { std::cerr << "covoxslam_cuda: " << queried.error().message << '\n'; return 1; }
        const auto& info = *queried;
        std::cout << info.name << ", device " << info.index << ", compute "
                  << info.compute_major << '.' << info.compute_minor << '\n'
                  << "runtime " << info.runtime_version << ", driver " << info.driver_version
                  << ", memory bytes " << info.total_memory_bytes << '\n';
        return 0;
}
