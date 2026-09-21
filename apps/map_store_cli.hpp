#pragma once
#include <covoxslam/core/byte_units.hpp>
#include <covoxslam/core/result.hpp>
#include <covoxslam/pipeline/mapping_session.hpp>
#include "cli_lookup.hpp"
#include <charconv>
#include <filesystem>
#include <string_view>
#include <system_error>

namespace covoxslam
{
enum class store_option { unknown, layout, fields, output, memory, mapping, negative_band };
enum class store_value { unknown, soa, copied, borrowed, resident, resident_soa };
inline constexpr cli_option_entry<store_option> store_options[]{
    {"--map-store", store_option::layout}, {"--pgo-fields", store_option::fields},
    {"--store-output", store_option::output}, {"--store-memory-mib", store_option::memory},
    {"--mapping-storage", store_option::mapping}, {"--store-negative-band", store_option::negative_band}
};
inline constexpr cli_option_entry<store_value> store_values[]{
    {"soa", store_value::soa},
    {"copied", store_value::copied}, {"borrowed", store_value::borrowed}, {"resident", store_value::resident},
    {"resident-soa", store_value::resident_soa}
};
struct store_selector_requirement
{
    unsigned mask{};
    std::string_view required_value;
};
inline store_selector_requirement required_store_selector(std::string_view key) noexcept
{
    switch (lookup_cli_option(key, store_options)) {
    case store_option::mapping: return {1, "resident-soa"};
    case store_option::layout: return {2, "soa"};
    default: return {};
    }
}
inline operation_result<bool> parse_store_option(std::string_view key, std::string_view value, store_pipeline_options& result)
{
    switch (lookup_cli_option(key, store_options)) {
    case store_option::negative_band:
        if (value != "true" && value != "false")
            return std::unexpected(operation_error{"store negative band must be true or false"});
        result.negative_band = value == "true";
        return true;
    case store_option::mapping:
        switch (lookup_cli_option(value, store_values)) {
        case store_value::resident_soa: result.resident_mapping = true; break;
        default: return std::unexpected(operation_error{"coVoxSLAM requires resident-soa mapping storage"});
        }
        return true;
    case store_option::layout:
        result.enabled = true;
        switch (lookup_cli_option(value, store_values)) {
        case store_value::soa: result.layout = storage::layout::pool_soa; break;
        default: return std::unexpected(operation_error{"coVoxSLAM requires soa map storage"});
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
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !amount)
            return std::unexpected(operation_error{"store memory must be a positive MiB count"});
        auto bytes = byte_count(amount, byte_unit::mebibyte);
        if (!bytes) return std::unexpected(operation_error{"store memory must be a positive MiB count"});
        result.cpu_bytes = result.gpu_bytes = *bytes;
        return true;
    }
    default: return false;
    }
}
inline operation_status validate_store_output_paths(
    const store_pipeline_options& store, const std::filesystem::path& export_path)
{
    if (store.checkpoint_root.empty() || export_path.empty()) return {};
    std::error_code error;
    const auto store_path = std::filesystem::weakly_canonical(store.checkpoint_root, error);
    if (error) return std::unexpected(operation_error{"checkpoint path is invalid"});
    const auto canonical_export = std::filesystem::weakly_canonical(export_path, error);

    if (error) return std::unexpected(operation_error{"export path is invalid"});
    if (store_path == canonical_export)
        return std::unexpected(operation_error{"checkpoint and export paths must differ"});
    return {};
}
}
