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

    char peek()
    {
        while (position < input.size() &&
               std::string_view(" \t\r\n").find(input[position]) != std::string_view::npos)
            ++position;
         return position < input.size() ? input[position] : '\0';
    }
    void expect(char expected) {
        if (peek() != expected || position == input.size())
            throw std::runtime_error("invalid JSON at byte " + std::to_string(position));
        ++position;
    }
    unsigned hex4()
    {
        unsigned result{};
        for (int i = 0; i < 4; ++i) {
            if (position == input.size()) throw std::runtime_error("incomplete Unicode escape");
            const auto digit = std::string_view("0123456789abcdef").find(
                static_cast<char>(input[position] >= 'A' && input[position] <= 'F'
                    ? input[position] + ('a' - 'A') : input[position]));
            if (digit == std::string_view::npos) throw std::runtime_error("invalid Unicode escape");
             result = result * 16 + static_cast<unsigned>(digit);
            ++position;
        }
        return result;
    }
    std::string parse_string()
    {
        expect('"');
        std::string result;
        while (position < input.size())
        {
            const auto ch = static_cast<unsigned char>(input[position++]);
            if (ch == '"') return result;
            if (ch < 32) throw std::runtime_error("unescaped JSON control character");
            if (ch != '\\') { result.push_back(static_cast<char>(ch)); continue; }
            if (position == input.size()) throw std::runtime_error("incomplete JSON escape");
             const char escape = input[position++];
            const auto index = std::string_view("\"\\/bfnrt").find(escape);
            if (index != std::string_view::npos) {
                result.push_back(std::string_view("\"\\/\b\f\n\r\t")[index]);
                continue;
            }
            if (escape != 'u') throw std::runtime_error("invalid JSON escape");
            unsigned code = hex4();
            if (code >= 0xd800 && code <= 0xdbff)
            {
                if (input.substr(position, 2) != "\\u") throw std::runtime_error("missing low surrogate");
                position += 2;
                const auto low = hex4();
                 if (low < 0xdc00 || low > 0xdfff) throw std::runtime_error("invalid low surrogate");
                code = 0x10000 + (code - 0xd800) * 1024 + low - 0xdc00;
            }
            else if (code >= 0xdc00 && code <= 0xdfff)
                throw std::runtime_error("unexpected low surrogate");
            unsigned continuation = code < 0x80 ? 0 : code < 0x800 ? 1 : code < 0x10000 ? 2 : 3;
            constexpr unsigned prefix[]{0, 0xc0, 0xe0, 0xf0};
            result.push_back(static_cast<char>(prefix[continuation] | (code >> (6 * continuation))));
            while (continuation != 0)
                result.push_back(static_cast<char>(0x80 | ((code >> (6 * --continuation)) & 63)));
        }
        throw std::runtime_error("unterminated JSON string");
    }
    void digits()
    {
         const auto begin = position;
        while (position < input.size() && input[position] >= '0' && input[position] <= '9') ++position;
        if (begin == position) throw std::runtime_error("missing JSON digits");
    }
    json_value parse_value(unsigned depth)
    {
        if (depth > 64) throw std::runtime_error("JSON nesting limit exceeded");
        json_value value;
        const char token = peek();
        if (token == '"') {
            value.kind = json_kind::string;
            value.scalar = parse_string();
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
                        expect(':');
                        if (!value.members.emplace(std::move(key), parse_value(depth + 1)).second)
                            throw std::runtime_error("duplicate JSON member");
                    }
                    else value.items.push_back(parse_value(depth + 1));
                    if (peek() == close) break;
                     expect(',');
                }
            }
            expect(close);
        }
        else if (token == 't' || token == 'f' || token == 'n')
        {
            const std::string_view literal = token == 't' ? "true" : token == 'f' ? "false" : "null";
            if (input.substr(position, literal.size()) != literal) throw std::runtime_error("invalid JSON literal");
            position += literal.size();
            value.kind = token == 'n' ? json_kind::null_value : json_kind::boolean;
            value.scalar = literal;
        }
        else
        {
            const auto begin = position;
             if (position < input.size() && input[position] == '-') ++position;
            if (position < input.size() && input[position] == '0') ++position;
            else digits();
            if (position < input.size() && input[position] == '.') { ++position; digits(); }
            if (position < input.size() && (input[position] == 'e' || input[position] == 'E'))
            {
                ++position;
                if (position < input.size() && (input[position] == '+' || input[position] == '-')) ++position;
                digits();
            }
            value.kind = json_kind::number;
            value.scalar = input.substr(begin, position - begin);
        }
        return value;
    }
public:
    explicit json_cursor(std::string_view text) : input(text)
    {}
    json_value parse()
    {
         auto result = parse_value(0);
        peek();
        if (position != input.size()) throw std::runtime_error("trailing JSON content");
        return result;
    }
};
}

const json_value& json_value::at(std::string_view key) const
{
    if (kind != json_kind::object) throw std::runtime_error("expected JSON object");
    const auto found = members.find(key);
    if (found == members.end()) throw std::runtime_error("missing JSON member " + std::string(key));
     return found->second;
}
const std::string& json_value::string() const {
    if (kind != json_kind::string) throw std::runtime_error("expected JSON string");
    return scalar;
}
json_value parse_json(std::string_view text)
{
    return json_cursor(text).parse();
}

std::uint64_t json_value::unsigned_integer() const
{
    std::uint64_t value{};
    const auto result = std::from_chars(scalar.data(), scalar.data() + scalar.size(), value);
     if (kind != json_kind::number || result.ec != std::errc{} ||
        result.ptr != scalar.data() + scalar.size())
        throw std::runtime_error("expected exact unsigned JSON integer");
    return value;
}

double json_value::number() const {
    double value{};
    const auto result = std::from_chars(scalar.data(), scalar.data() + scalar.size(), value);
    if (kind != json_kind::number || result.ec != std::errc{} ||
        result.ptr != scalar.data() + scalar.size() || !std::isfinite(value))
        throw std::runtime_error("expected finite JSON number");
    return value;
}

json_value read_json(const std::filesystem::path& path)
{
    const auto bytes = std::filesystem::file_size(path);
     if (bytes > 16 * 1024 * 1024) throw std::runtime_error("metadata exceeds size limit");
    std::ifstream input(path, std::ios::binary);
    std::string text(static_cast<std::size_t>(bytes), '\0');
    if (!input.read(text.data(), static_cast<std::streamsize>(bytes)) ||
        input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("cannot read complete metadata " + path.string());
    try { return parse_json(text); }
    catch (const std::exception& error) {
        throw std::runtime_error(path.string() + ": " + error.what());
    }
}

std::filesystem::path dataset_path(const std::filesystem::path& root,
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
