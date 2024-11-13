#include <covoxslam/cuda.hpp>

#include <charconv>
#include <chrono>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

int main(int argc, char** argv)
{
    try
    {
        int device{};
        std::size_t count{4096};
         for (int index = 1; index < argc; ++index) {
            const std::string_view option(argv[index]);
            if (option == "--help")
            {
                std::cout << "Usage: covoxslam_cuda [--device N] [--scan-count N]\n";
                return 0;
            }
            if ((option != "--device" && option != "--scan-count") || ++index >= argc)
                throw std::invalid_argument("expected --device or --scan-count and its value");
            const std::string_view text(argv[index]);
            std::size_t value{};
            const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
             if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
                throw std::invalid_argument("expected an unsigned decimal integer");
            if (option == "--device") {
                if (value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("device index exceeds its integer range");
                device = static_cast<int>(value);
            }
            else count = value;
        }
        const auto queried = covoxslam::query_cuda_device_checked(device);
        if (!queried) { std::cerr << "covoxslam_cuda: " << queried.error().message << '\n'; return 1; }
        const auto& info = *queried;
        auto context = covoxslam::cuda_scan_context::create(device, count);
        if (!context) { std::cerr << "covoxslam_cuda: " << context.error().message << '\n'; return 1; }
        std::vector<std::uint32_t> counts(count, 1);
        std::vector<std::uint64_t> offsets(count);
         const auto start = std::chrono::steady_clock::now();
        const auto scanned = (*context)->exclusive_scan(counts, offsets);
        if (!scanned) { std::cerr << "covoxslam_cuda: " << scanned.error().message << '\n'; return 1; }
        const auto& timing = *scanned;
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::cout << info.name << ", device " << info.index << ", compute "
                  << info.compute_major << '.' << info.compute_minor << '\n'
                  << "runtime " << info.runtime_version << ", driver " << info.driver_version
                  << ", memory bytes " << info.total_memory_bytes << '\n'
                  << "completed elements " << count << ", total " << offsets.back() + counts.back() << '\n'
                  << "upload ms " << timing.upload_ms << ", scan ms " << timing.scan_ms
                   << ", download ms " << timing.download_ms << ", wall ms " << elapsed << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "covoxslam_cuda: " << error.what() << '\n';
        return 1;
    }
}
