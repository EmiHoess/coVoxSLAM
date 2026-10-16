#include "dataset_metadata.hpp"
#include "input_limits.hpp"
#include <charconv>
#include <cmath>

namespace covoxslam::detail
{
operation_error toml_cursor::error(std::string_view message) const
{
    return {"TOML byte " + std::to_string(position) + ": " + std::string(message)};
}
char toml_cursor::peek() const noexcept { return position < input.size() ? input[position] : '\0'; }
void toml_cursor::space(bool multiline)
{
    while (position < input.size()) {
        const auto ch = peek();
        if (ch == ' ' || ch == '\t' || ch == '\r' || (multiline && ch == '\n')) ++position;
        else if (multiline && ch == '#') {
            while (position < input.size() && peek() != '\n') ++position;
        } else break;
    }
}
operation_result<std::string> toml_cursor::string()
{
    const auto quote = peek();
    if (quote != '\"' && quote != '\'') return std::unexpected(error("expected quoted string"));
    ++position;
    std::string result;

    while (position < input.size()) {
        const auto ch = static_cast<unsigned char>(input[position++]);
        if (ch == quote) return result;

        if ((ch < 32 && ch != '\t') || ch == 127)
            return std::unexpected(error("control character in string"));
        if (ch != '\\' || quote == '\'') { result.push_back(static_cast<char>(ch)); continue; }
        if (position == input.size()) return std::unexpected(error("incomplete escape"));
        const auto escaped = input[position++];
        const auto offset = std::string_view("\"\\bfnrt").find(escaped);
        if (offset != std::string_view::npos) {
             result.push_back(std::string_view("\"\\\b\f\n\r\t")[offset]);
            continue;
        }
        if (escaped != 'u' && escaped != 'U') return std::unexpected(error("unsupported string escape"));
        const unsigned digits = escaped == 'u' ? 4 : 8;
        std::uint32_t code{};

        for (unsigned index = 0; index < digits; ++index) {
            if (position == input.size()) return std::unexpected(error("incomplete Unicode escape"));
            auto digit = input[position++];
            if (digit >= 'A' && digit <= 'F') digit = static_cast<char>(digit + 32);
            const auto n = std::string_view("0123456789abcdef").find(digit);

            if (n == std::string_view::npos) return std::unexpected(error("invalid Unicode escape"));
            code = code * 16 + static_cast<unsigned>(n);
        }
        if (code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
            return std::unexpected(error("invalid Unicode scalar"));
        if (code < 0x80) result.push_back(static_cast<char>(code));
        else {
            const unsigned trailing = code < 0x800 ? 1 : code < 0x10000 ? 2 : 3;
            result.push_back(static_cast<char>((0xff << (7 - trailing)) | (code >> (6 * trailing))));

            for (unsigned n = trailing; n > 0; --n)
                result.push_back(static_cast<char>(0x80 | ((code >> (6 * (n - 1))) & 63)));
        }
    }
    return std::unexpected(error("unterminated string"));
}
operation_result<std::vector<std::string>> toml_cursor::keys()
{
    std::vector<std::string> result;
    do {
        space();
        if (result.size() >= maximum_toml_key_components) return std::unexpected(error("key path exceeds limit"));
        if (peek() == '\"' || peek() == '\'') {
            auto key = string();
            if (!key) return std::unexpected(key.error());
            result.push_back(std::move(*key));
        } else {
            const auto start = position;
            while (position < input.size()) {
                const auto ch = peek();
                if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
                    (ch >= '0' && ch <= '9') || ch == '_' || ch == '-')) break;
                ++position;
            }
            if (start == position) return std::unexpected(error("expected key"));
            result.emplace_back(input.substr(start, position - start));
        }
        space();
        if (peek() != '.') return result;
        ++position;
    } while (position < input.size());
    return std::unexpected(error("incomplete dotted key"));
}
operation_result<json_value> toml_cursor::atom()
{
    const auto start = position;
    while (position < input.size() && std::string_view(" \t\r\n,]}#").find(peek()) == std::string_view::npos)
        ++position;
    auto token = input.substr(start, position - start);
    json_value result;
    result.scalar = token;

    if (token == "true" || token == "false") { result.kind = json_kind::boolean; return result; }
    result.kind = json_kind::number;
    if (token.starts_with('+')) token.remove_prefix(1);
    if (token.empty()) return std::unexpected(error("expected decimal value"));
    auto magnitude = token;
    if (magnitude.starts_with('-')) magnitude.remove_prefix(1);
    const auto decimal_end = magnitude.find_first_of(".eE");
    const auto whole = magnitude.substr(0, decimal_end);
    if (whole.empty() || (whole.size() > 1 && whole.front() == '0') ||
        whole.find_first_not_of("0123456789") != std::string_view::npos)
        return std::unexpected(error("invalid decimal integer part"));
    if (decimal_end != std::string_view::npos) {
        auto remaining = magnitude.substr(decimal_end);
        if (remaining.front() == '.') {
            remaining.remove_prefix(1);
            const auto exponent = remaining.find_first_of("eE");
            const auto fraction = remaining.substr(0, exponent);

            if (fraction.empty() || fraction.find_first_not_of("0123456789") != std::string_view::npos)
                return std::unexpected(error("invalid fraction"));
            remaining = exponent == std::string_view::npos ? std::string_view{} : remaining.substr(exponent);
        }
        if (!remaining.empty()) {
            remaining.remove_prefix(1);
            if (remaining.starts_with('+') || remaining.starts_with('-')) remaining.remove_prefix(1);
            if (remaining.empty() || remaining.find_first_not_of("0123456789") != std::string_view::npos)
                return std::unexpected(error("invalid exponent"));
        }
    }
    if (token.find_first_of(".eE") == std::string_view::npos) {
        std::int64_t number{};
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
            return std::unexpected(error("expected signed 64 bit decimal integer"));
         result.scalar = std::to_string(number);
    } else {
        double number{};
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), number);

        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || !std::isfinite(number))
            return std::unexpected(error("expected finite decimal number"));
        result.scalar = token;
    }
    return result;
}
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
    auto* node = &parent;
    for (const auto& key : prefix) {
        node = &node->members.find(key)->second;
        if (node->kind == json_kind::array) node = &node->items.back();
        node->metadata_defined = true;
    }
    return {};
}
operation_result<json_value> toml_cursor::value(unsigned depth)
{
    if (depth > maximum_toml_nesting) return std::unexpected(error("metadata nesting exceeds limit"));
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
    if (input.size() > maximum_metadata_bytes || input.find('\0') != std::string_view::npos)
        return std::unexpected(error("invalid metadata size or null byte"));
    for (std::size_t offset = 0; offset < input.size();) {
        const auto leading = static_cast<unsigned char>(input[offset++]);
        if (leading < 128) {
            if (leading == '\r' && (offset == input.size() || input[offset] != '\n'))
                return std::unexpected(error("carriage return requires newline"));
            continue;
        }
        const unsigned count = leading >= 0xf0 && leading <= 0xf4 ? 3 :
            leading >= 0xe0 && leading <= 0xef ? 2 : leading >= 0xc2 && leading <= 0xdf ? 1 : 0;
        if (!count || count > input.size() - offset) return std::unexpected(error("invalid UTF8 sequence"));
        std::uint32_t code = leading & ((1u << (6 - count)) - 1);

        for (unsigned index = 0; index < count; ++index) {
            const auto next = static_cast<unsigned char>(input[offset++]);
            if ((next & 0xc0) != 0x80) return std::unexpected(error("invalid UTF8 continuation"));
            code = (code << 6) | (next & 63);
        }
        if (code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff) ||
            code < (count == 1 ? 0x80u : count == 2 ? 0x800u : 0x10000u))
            return std::unexpected(error("invalid UTF8 scalar"));
    }
    if (input.starts_with("\xef\xbb\xbf")) position = 3;
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
