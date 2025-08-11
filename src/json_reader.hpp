#pragma once

#include <covoxslam/core/geometry.hpp>
#include <covoxslam/core/result.hpp>

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
    bool metadata_defined{}, metadata_sealed{};

    const json_value& at(std::string_view key) const;
    const std::string& string() const;
    std::uint64_t unsigned_integer() const;
    double number() const;
}
;

operation_result<json_value> parse_json(std::string_view text);
operation_result<json_value> read_json(const std::filesystem::path& path);
pose read_pose(const json_value& value);
std::filesystem::path dataset_path(const std::filesystem::path& root,
                                   const std::filesystem::path& relative);
}
