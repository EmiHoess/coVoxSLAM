#include "json_reader.hpp"

#include <stdexcept>
#include <charconv>
#include <cmath>
#include <fstream>
#include <utility>

namespace covoxslam::detail
{
namespace
{
class json_cursor
{
    std::string_view input;
    std::size_t position{};

    operation_error error(std::string_view message) const
    {
        return {std::string(message) + " at byte " + std::to_string(position)};
    }

    char peek()
    {
        while (position < input.size() &&
               std::string_view(" \t\r\n").find(input[position]) != std::string_view::npos)
            ++position;
         return position < input.size() ? input[position] : '\0';
    }
    operation_status expect(char expected) {
        if (peek() != expected || position == input.size())
            return std::unexpected(error("invalid JSON"));
        ++position;
        return {};
    }
    operation_result<unsigned> hex4()
    {
        unsigned result{};
        for (int i = 0; i < 4; ++i) {
            if (position == input.size()) return std::unexpected(error("incomplete Unicode escape"));
            const auto digit = std::string_view("0123456789abcdef").find(
                static_cast<char>(input[position] >= 'A' && input[position] <= 'F'
                    ? input[position] + ('a' - 'A') : input[position]));
            if (digit == std::string_view::npos) return std::unexpected(error("invalid Unicode escape"));
             result = result * 16 + static_cast<unsigned>(digit);
            ++position;
        }
        return result;
    }
    operation_result<std::string> parse_string()
    {
        auto quote = expect('"');
        if (!quote) return std::unexpected(quote.error());
        std::string result;
        while (position < input.size())
        {
            const auto ch = static_cast<unsigned char>(input[position++]);
            if (ch == '"') return result;
            if (ch < 32) return std::unexpected(error("unescaped JSON control character"));
            if (ch != '\\') { result.push_back(static_cast<char>(ch)); continue; }
            if (position == input.size()) return std::unexpected(error("incomplete JSON escape"));
             const char escape = input[position++];
            const auto index = std::string_view("\"\\/bfnrt").find(escape);
            if (index != std::string_view::npos) {
                result.push_back(std::string_view("\"\\/\b\f\n\r\t")[index]);
                continue;
            }
            if (escape != 'u') return std::unexpected(error("invalid JSON escape"));
            auto parsed_code = hex4();
            if (!parsed_code) return std::unexpected(parsed_code.error());
            unsigned code = *parsed_code;
            if (code >= 0xd800 && code <= 0xdbff)
            {
                if (input.substr(position, 2) != "\\u") return std::unexpected(error("missing low surrogate"));
                position += 2;
                const auto parsed_low = hex4();
                if (!parsed_low) return std::unexpected(parsed_low.error());
                 if (*parsed_low < 0xdc00 || *parsed_low > 0xdfff)
                    return std::unexpected(error("invalid low surrogate"));
                code = 0x10000 + (code - 0xd800) * 1024 + *parsed_low - 0xdc00;
            }
            else if (code >= 0xdc00 && code <= 0xdfff)
                return std::unexpected(error("unexpected low surrogate"));
            unsigned continuation = code < 0x80 ? 0 : code < 0x800 ? 1 : code < 0x10000 ? 2 : 3;
            constexpr unsigned prefix[]{0, 0xc0, 0xe0, 0xf0};
            result.push_back(static_cast<char>(prefix[continuation] | (code >> (6 * continuation))));
            while (continuation != 0)
                result.push_back(static_cast<char>(0x80 | ((code >> (6 * --continuation)) & 63)));
        }
        return std::unexpected(error("unterminated JSON string"));
    }
    operation_status digits()
    {
         const auto begin = position;
        while (position < input.size() && input[position] >= '0' && input[position] <= '9') ++position;
        if (begin == position) return std::unexpected(error("missing JSON digits"));
        return {};
    }
    operation_result<json_value> parse_value(unsigned depth)
    {
        if (depth > 64) return std::unexpected(error("JSON nesting limit exceeded"));
        json_value value;
        const char token = peek();
        if (token == '"') {
            value.kind = json_kind::string;
            auto parsed = parse_string();
            if (!parsed) return std::unexpected(parsed.error());
            value.scalar = std::move(*parsed);
        }
        else if (token == '{' || token == '[')
        {
            value.kind = token == '{' ? json_kind::object : json_kind::array;
             const char close = token == '{' ? '}' : ']';
            ++position;
            if (peek() != close)
            {
                for (;;)
                {
                    if (token == '{') {
                        auto key = parse_string();
                        if (!key) return std::unexpected(key.error());
                        auto separator = expect(':');
                        if (!separator) return std::unexpected(separator.error());
                        auto parsed = parse_value(depth + 1);
                        if (!parsed) return std::unexpected(parsed.error());
                        if (!value.members.emplace(std::move(*key), std::move(*parsed)).second)
                            return std::unexpected(error("duplicate JSON member"));
                    }
                    else {
                        auto parsed = parse_value(depth + 1);
                        if (!parsed) return std::unexpected(parsed.error());
                        value.items.push_back(std::move(*parsed));
                    }
                    if (peek() == close) break;
                    auto separator = expect(',');
                    if (!separator) return std::unexpected(separator.error());
                }
            }
            auto closed = expect(close);
            if (!closed) return std::unexpected(closed.error());
        }
        else if (token == 't' || token == 'f' || token == 'n')
        {
            const std::string_view literal = token == 't' ? "true" : token == 'f' ? "false" : "null";
            if (input.substr(position, literal.size()) != literal)
                return std::unexpected(error("invalid JSON literal"));
            position += literal.size();
            value.kind = token == 'n' ? json_kind::null_value : json_kind::boolean;
            value.scalar = literal;
        }
        else
        {
            const auto begin = position;
             if (position < input.size() && input[position] == '-') ++position;
            if (position < input.size() && input[position] == '0') ++position;
            else {
                auto integral = digits();
                if (!integral) return std::unexpected(integral.error());
            }
            if (position < input.size() && input[position] == '.') {
                ++position;
                auto fraction = digits();
                if (!fraction) return std::unexpected(fraction.error());
            }
            if (position < input.size() && (input[position] == 'e' || input[position] == 'E'))
            {
                ++position;
                if (position < input.size() && (input[position] == '+' || input[position] == '-')) ++position;
                auto exponent = digits();
                if (!exponent) return std::unexpected(exponent.error());
            }
            value.kind = json_kind::number;
            value.scalar = input.substr(begin, position - begin);
        }
        return value;
    }
public:
    explicit json_cursor(std::string_view text) : input(text)
    {}
    operation_result<json_value> parse()
    {
         auto result = parse_value(0);
        if (!result) return std::unexpected(result.error());
        peek();
        if (position != input.size()) return std::unexpected(error("trailing JSON content"));
        return result;
    }
};
}

operation_result<std::reference_wrapper<const json_value>> json_value::member(std::string_view key) const
{
    if (kind != json_kind::object) return std::unexpected(operation_error{"expected JSON object"});
    const auto found = members.find(key);
    if (found == members.end())
        return std::unexpected(operation_error{"missing JSON member " + std::string(key)});
     return std::cref(found->second);
}
operation_result<std::reference_wrapper<const std::string>> json_value::string() const
{
    if (kind != json_kind::string) return std::unexpected(operation_error{"expected JSON string"});
    return std::cref(scalar);
}
operation_result<std::uint64_t> json_value::unsigned_integer() const
{
    if (kind != json_kind::number)
        return std::unexpected(operation_error{"expected exact unsigned JSON integer"});
    std::uint64_t value{};
    const auto result = std::from_chars(scalar.data(), scalar.data() + scalar.size(), value);
     if (result.ec != std::errc{} || result.ptr != scalar.data() + scalar.size())
        return std::unexpected(operation_error{"expected exact unsigned JSON integer"});
    return value;
}
operation_result<double> json_value::number() const
{
    if (kind != json_kind::number)
        return std::unexpected(operation_error{"expected finite JSON number"});
    double value{};
    const auto result = std::from_chars(scalar.data(), scalar.data() + scalar.size(), value);
    if (result.ec != std::errc{} || result.ptr != scalar.data() + scalar.size() || !std::isfinite(value))
        return std::unexpected(operation_error{"expected finite JSON number"});
    return value;
}

const json_value& json_value::legacy_at(std::string_view key) const
{
    if (kind != json_kind::object) throw std::runtime_error("expected JSON object");
    const auto found = members.find(key);
    if (found == members.end()) throw std::runtime_error("missing JSON member " + std::string(key));
     return found->second;
}
const std::string& json_value::legacy_string() const {
    if (kind != json_kind::string) throw std::runtime_error("expected JSON string");
    return scalar;
}
operation_result<json_value> parse_json(std::string_view text)
{
    return json_cursor(text).parse();
}

std::uint64_t json_value::legacy_unsigned_integer() const
{
    std::uint64_t value{};
    const auto result = std::from_chars(scalar.data(), scalar.data() + scalar.size(), value);
     if (kind != json_kind::number || result.ec != std::errc{} ||
        result.ptr != scalar.data() + scalar.size())
        throw std::runtime_error("expected exact unsigned JSON integer");
    return value;
}

double json_value::legacy_number() const {
    double value{};
    const auto result = std::from_chars(scalar.data(), scalar.data() + scalar.size(), value);
    if (kind != json_kind::number || result.ec != std::errc{} ||
        result.ptr != scalar.data() + scalar.size() || !std::isfinite(value))
        throw std::runtime_error("expected finite JSON number");
    return value;
}

operation_result<json_value> read_json(const std::filesystem::path& path)
{
    std::error_code error;
    const auto bytes = std::filesystem::file_size(path, error);
    if (error) return std::unexpected(operation_error{
        "cannot inspect metadata " + path.string() + ": " + error.message()});
     if (bytes > 16 * 1024 * 1024)
        return std::unexpected(operation_error{"metadata exceeds size limit " + path.string()});
    std::ifstream input(path, std::ios::binary);
    std::string text(static_cast<std::size_t>(bytes), '\0');
    if (!input.read(text.data(), static_cast<std::streamsize>(bytes)) ||
        input.peek() != std::char_traits<char>::eof() || input.bad())
        return std::unexpected(operation_error{"cannot read complete metadata " + path.string()});
    auto result = parse_json(text);
    if (!result) result.error().message = path.string() + ": " + result.error().message;
    return result;
}

operation_result<std::filesystem::path> dataset_path(const std::filesystem::path& root,
                                                     const std::filesystem::path& relative)
{
    if (relative.empty() || relative.has_root_path())
        return std::unexpected(operation_error{"expected relative dataset path"});
    for (const auto& component : relative)
        if (component == ".." || component.native().find(
            static_cast<std::filesystem::path::value_type>(':')) != std::filesystem::path::string_type::npos)
            return std::unexpected(operation_error{"invalid dataset path component"});
    std::error_code error;
    const auto base = std::filesystem::canonical(root, error);
    if (error) return std::unexpected(operation_error{"cannot resolve dataset root"});
    error.clear();
     const auto target = std::filesystem::weakly_canonical(base / relative, error);
    if (error) return std::unexpected(operation_error{"cannot resolve dataset path"});
    const auto local = target.lexically_relative(base);
    if (local.empty() || *local.begin() == "..")
        return std::unexpected(operation_error{"dataset path escapes root"});
    return target;
}

std::filesystem::path legacy_dataset_path(const std::filesystem::path& root,
                                          const std::filesystem::path& relative)
{
    if (relative.empty() || relative.has_root_path())
        throw std::runtime_error("expected relative dataset path");
    for (const auto& component : relative)
        if (component == ".." || component.string().find(':') != std::string::npos)
            throw std::runtime_error("invalid dataset path component");
    const auto base = std::filesystem::canonical(root);
     const auto target = std::filesystem::weakly_canonical(base / relative);
    const auto local = target.lexically_relative(base);
    if (local.empty() || *local.begin() == "..")
        throw std::runtime_error("dataset path escapes root");
    return target;
}
}
