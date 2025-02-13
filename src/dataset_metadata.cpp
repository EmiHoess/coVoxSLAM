#include "dataset_metadata.hpp"
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
        if (result.size() >= 64) return std::unexpected(error("key path exceeds limit"));
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
}
