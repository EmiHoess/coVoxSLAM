#pragma once
#include <covoxslam/store_pipeline.hpp>
#include "cli_lookup.hpp"
#include <charconv>
#include <limits>

namespace covoxslam
{
enum class store_option { unknown, layout, fields, output, memory };
enum class store_value { unknown, none, aos, soa, aosoa, copied, borrowed, resident };
inline constexpr cli_option_entry<store_option> store_options[]{
    {"--map-store", store_option::layout}, {"--pgo-fields", store_option::fields},
    {"--store-output", store_option::output}, {"--store-memory-mib", store_option::memory}
};
inline constexpr cli_option_entry<store_value> store_values[]{
    {"none", store_value::none}, {"aos", store_value::aos}, {"soa", store_value::soa}, {"aosoa", store_value::aosoa},
    {"copied", store_value::copied}, {"borrowed", store_value::borrowed}, {"resident", store_value::resident}
};
inline operation_result<bool> parse_store_option(std::string_view key, std::string_view value, store_pipeline_options& result)
{
    switch (lookup_cli_option(key, store_options)) {
    case store_option::layout:
        result.enabled = true;
        switch (lookup_cli_option(value, store_values)) {
        case store_value::none: result.enabled = false; break;
        case store_value::aos: result.layout = storage::layout::pooled_aos; break;
        case store_value::soa: result.layout = storage::layout::pool_soa; break;
        case store_value::aosoa: result.layout = storage::layout::tile_aosoa; break;
        default: return std::unexpected(operation_error{"map store must be none, aos, soa or aosoa"});
        }
        return true;
    case store_option::fields:
        switch (lookup_cli_option(value, store_values)) {
        case store_value::copied: result.fields = stored_pgo_fields::copied; break;
        case store_value::borrowed: result.fields = stored_pgo_fields::borrowed; break;
        case store_value::resident: result.fields = stored_pgo_fields::resident; break;
        default: return std::unexpected(operation_error{"PGO fields must be copied, borrowed or resident"});
        }
        return true;
    case store_option::output:
        if (value.empty()) return std::unexpected(operation_error{"store output is empty"});
        result.checkpoint_root = value;
        return true;
    case store_option::memory: {
        std::size_t amount{};
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), amount);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !amount ||
            amount > std::numeric_limits<std::size_t>::max() / 1048576)
            return std::unexpected(operation_error{"store memory must be a positive MiB count"});
        result.cpu_bytes = result.gpu_bytes = amount * 1048576;
        return true;
    }
    default: return false;
    }
}
}
