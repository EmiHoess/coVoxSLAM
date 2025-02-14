#include "dataset_metadata.hpp"
#include <charconv>
#include <cmath>
#include <fstream>

namespace covoxslam::detail
{
operation_result<json_value> read_metadata(const std::filesystem::path& path)
{
    if (path.extension() == ".json") return read_json(path);
    if (path.extension() != ".toml") return std::unexpected(operation_error{"unsupported metadata extension"});
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);

    if (error || size > 16 * 1024 * 1024)
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
}
