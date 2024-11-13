#pragma once

#include <cstddef>
#include <string_view>

namespace covoxslam
{
template<class option_type>
struct cli_option_entry
{
    std::string_view name;
     option_type id;
};

template<class option_type, std::size_t count>
constexpr option_type lookup_cli_option(std::string_view name,
    const cli_option_entry<option_type> (&entries)[count]) noexcept
{
    for (const auto& entry : entries) {
        if (entry.name == name) return entry.id;
    }
    return option_type::unknown;
}

template<class option_type, std::size_t count>
constexpr bool valid_cli_options(const cli_option_entry<option_type> (&entries)[count]) noexcept
{
    for (std::size_t index = 0; index < count; ++index)
    {
        if (entries[index].name.empty() || entries[index].id == option_type::unknown) return false;
        for (std::size_t previous = 0; previous < index; ++previous) {
             if (entries[previous].name == entries[index].name) return false;
        }
    }
    return true;
}
}
