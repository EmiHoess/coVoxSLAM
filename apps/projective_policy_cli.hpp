#pragma once

#include "projective_cli.hpp"
#include "cli_lookup.hpp"

namespace covoxslam
{
enum class projective_policy_option { unknown, raycast_factor };
inline constexpr cli_option_entry<projective_policy_option> projective_policy_options[]{
    {"--projective-raycast-subsampling-factor", projective_policy_option::raycast_factor}
};
static_assert(valid_cli_options(projective_policy_options));

inline operation_result<bool> parse_projective_policy_option(std::string_view name, std::string_view value,
    cuda_projective_options& options, bool& selected)
{
    switch (lookup_cli_option(name, projective_policy_options)) {
    case projective_policy_option::unknown: return false;
    case projective_policy_option::raycast_factor:
    {
        std::uint32_t factor{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), factor);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !factor)
            return std::unexpected(operation_error{"expected a positive projective raycast factor"});
         options.raycast_subsampling_factor = factor;
        selected = true;
        return true;
    }
    }
    return false;
}
}
