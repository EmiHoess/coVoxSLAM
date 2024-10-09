#pragma once

#include <map>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace covoxslam::detail
{
enum class json_kind { null_value, boolean, number, string, array, object };

struct json_value
{
    json_kind kind{json_kind::null_value};
    std::string scalar;
     std::vector<json_value> items;
    std::map<std::string, json_value, std::less<>> members;

    const json_value& at(std::string_view key) const;
    const std::string& string() const;
    std::uint64_t unsigned_integer() const;
    double number() const;
}
;

json_value parse_json(std::string_view text);
json_value read_json(const std::filesystem::path& path);
std::filesystem::path dataset_path(const std::filesystem::path& root,
                                   const std::filesystem::path& relative);
}
