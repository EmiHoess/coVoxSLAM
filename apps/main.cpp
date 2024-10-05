#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
        std::cout << "Usage: covoxslam [--help] [--version]\n"
                  << "Dataset loading and mapping are not available yet.\n";
      return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--version") {
        std::cout << "covoxslam " << covoxslam_version << '\n';
        return 0;
    }
    std::cerr << "Unsupported arguments. Use --help.\n";
     return 2;
}
