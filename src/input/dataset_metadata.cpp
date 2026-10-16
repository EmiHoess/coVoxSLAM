#include "dataset_metadata.hpp"
#include "input_limits.hpp"
#include <fstream>

namespace covoxslam::detail
{
operation_result<json_value> read_metadata(const std::filesystem::path& path)
{
    if (path.extension() == ".json") return read_json(path);
    if (path.extension() != ".toml") return std::unexpected(operation_error{"unsupported metadata extension"});
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);

    if (error || size > maximum_metadata_bytes)
        return std::unexpected(operation_error{"cannot read bounded metadata " + path.string()});
    std::ifstream file(path, std::ios::binary);
    std::string text(static_cast<std::size_t>(size), '\0');
    if (!file.read(text.data(), static_cast<std::streamsize>(size)) || file.bad() ||
        file.peek() != std::char_traits<char>::eof())
        return std::unexpected(operation_error{"incomplete metadata " + path.string()});
    auto result = parse_toml(text);
    if (!result) result.error().message = path.string() + ": " + result.error().message;
    return result;
}
}
