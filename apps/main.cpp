#include <iostream>
#include <covoxslam/dataset.hpp>
#include <string_view>

int main(int argc, char** argv) {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        std::cout << "Usage: covoxslam [--help] [--version] [--dataset directory]\n";
      return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "covoxslam " << covoxslam_version << '\n';
        return 0;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--dataset")
    {
        try
        {
            const auto data = covoxslam::load_dataset(argv[2]);
            std::cout << data.name << ": " << data.frames.size() << " frames, "
                      << data.sensor.rings << 'x' << data.sensor.columns << " organized LiDAR\n";
            return 0;
        }
        catch (const std::exception& error) {
            std::cerr << "covoxslam: " << error.what() << '\n';
            return 1;
        }
    }
    std::cerr << "Unsupported arguments. Use --help.\n";
     return 2;
}
