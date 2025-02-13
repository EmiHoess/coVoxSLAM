#pragma once
#include "json_reader.hpp"
#include <covoxslam/result.hpp>
#include <set>

namespace covoxslam::detail
{
class toml_cursor
{
    std::string_view input;
    std::size_t position{};
    std::set<const json_value*> explicit_tables;
    std::set<const json_value*> sealed_tables;
    operation_error error(std::string_view message) const;
    void space(bool multiline = false);
    char peek() const noexcept;
    operation_result<std::string> string();
    operation_result<std::vector<std::string>> keys();
    operation_result<json_value> atom();
    operation_result<json_value> value(unsigned depth);
    operation_result<json_value*> table(json_value& parent, const std::vector<std::string>& path, bool array);
    operation_status assign(json_value& parent, const std::vector<std::string>& path, json_value item);
public:
    explicit toml_cursor(std::string_view text) : input(text) {}
    operation_result<json_value> parse();
};
operation_result<json_value> parse_toml(std::string_view text);
operation_result<json_value> read_metadata(const std::filesystem::path& path);
}
