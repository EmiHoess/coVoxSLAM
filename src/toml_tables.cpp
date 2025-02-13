#include "dataset_metadata.hpp"

namespace covoxslam::detail
{
operation_result<json_value*> toml_cursor::table(json_value& parent,
    const std::vector<std::string>& path, bool array)
{
    auto* node = &parent;

    for (std::size_t index = 0; index < path.size(); ++index) {
        if (node->kind != json_kind::object || node->metadata_sealed)
            return std::unexpected(error("cannot extend this value"));
        auto [found, inserted] = node->members.try_emplace(path[index]);
        node = &found->second;
        const bool last = index + 1 == path.size();

        if (inserted) node->kind = last && array ? json_kind::array : json_kind::object;
        if (node->kind == json_kind::array) {
            if (last && array) {
                if (!node->metadata_defined && !inserted)
                    return std::unexpected(error("cannot append a static array"));
                node->metadata_defined = true;
                node->items.emplace_back();
                node->items.back().kind = json_kind::object;
            }
            if (!node->metadata_defined || node->items.empty())
                return std::unexpected(error("expected table array"));
            node = &node->items.back();
        } else if (last && array) return std::unexpected(error("expected table array"));

        if (node->kind != json_kind::object || node->metadata_sealed)
            return std::unexpected(error("table collides with value"));
    }
     return node;
}
operation_status toml_cursor::assign(json_value& parent, const std::vector<std::string>& path, json_value item)
{
    if (path.empty()) return std::unexpected(error("empty assignment path"));
    const std::vector<std::string> prefix(path.begin(), path.end() - 1);
    auto destination = table(parent, prefix, false);

    if (!destination) return std::unexpected(destination.error());
    if ((*destination)->metadata_sealed || !(*destination)->members.emplace(path.back(), std::move(item)).second)
        return std::unexpected(error("duplicate key or sealed table"));
    return {};
}
operation_result<json_value> toml_cursor::value(unsigned depth)
{
    if (depth > 64) return std::unexpected(error("metadata nesting exceeds limit"));
    space();
    json_value result;
    if (peek() == '\"' || peek() == '\'') {
        auto parsed = string();
        if (!parsed) return std::unexpected(parsed.error());
        result.kind = json_kind::string; result.scalar = std::move(*parsed);
        return result;
    }
    const auto opening = peek();
    if (opening != '[' && opening != '{') return atom();
    ++position;
    result.kind = opening == '[' ? json_kind::array : json_kind::object;
    const char closing = opening == '[' ? ']' : '}';
    space(opening == '[');

    while (peek() != closing) {
        std::vector<std::string> path;
        if (opening == '{') {
            auto parsed = keys();
            if (!parsed) return std::unexpected(parsed.error());
            path = std::move(*parsed);
            if (peek() != '=') return std::unexpected(error("expected inline assignment"));
            ++position;
        }
        auto parsed = value(depth + 1);
        if (!parsed) return std::unexpected(parsed.error());

        if (opening == '[') result.items.push_back(std::move(*parsed));
        else {
            auto added = assign(result, path, std::move(*parsed));
            if (!added) return std::unexpected(added.error());
        }
        space(opening == '[');
        if (peek() == closing) break;
        if (peek() != ',') return std::unexpected(error("expected separator"));
        ++position;
         space(opening == '[');
        if (opening == '{' && peek() == closing)
            return std::unexpected(error("inline trailing comma is unsupported"));
    }
    ++position;
    result.metadata_sealed = opening == '{';
    return result;
}
operation_result<json_value> toml_cursor::parse()
{
    if (input.size() > 16 * 1024 * 1024 || input.find('\0') != std::string_view::npos)
        return std::unexpected(error("invalid metadata size or null byte"));
    json_value result;
    result.kind = json_kind::object;
    auto* current = &result;
    space(true);

    while (position < input.size()) {
        if (peek() == '[') {
            ++position;
            const bool array = peek() == '[';
            if (array) ++position;
            auto path = keys();
            if (!path) return std::unexpected(path.error());
            if (peek() != ']') return std::unexpected(error("expected table closing bracket"));
            ++position;
            if (array) {
                if (peek() != ']') return std::unexpected(error("expected array closing bracket"));
                ++position;
            }
            auto selected = table(result, *path, array);
            if (!selected) return std::unexpected(selected.error());
            if ((*selected)->metadata_defined)
                return std::unexpected(error("duplicate table"));
            (*selected)->metadata_defined = true;
            current = *selected;
        } else {
            auto path = keys();
            if (!path) return std::unexpected(path.error());
            if (peek() != '=') return std::unexpected(error("expected assignment"));
            ++position;
            auto parsed = value(0);
            if (!parsed) return std::unexpected(parsed.error());
            auto added = assign(*current, *path, std::move(*parsed));
            if (!added) return std::unexpected(added.error());
        }
        space();
        if (peek() == '#') {
            while (position < input.size() && peek() != '\n') ++position;
        }
        if (position != input.size() && peek() != '\n')
            return std::unexpected(error("trailing statement content"));
        space(true);
    }
    return result;
}
operation_result<json_value> parse_toml(std::string_view text)
{
    return toml_cursor(text).parse();
}
}
