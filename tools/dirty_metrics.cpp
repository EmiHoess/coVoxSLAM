#include "layout_cpu.hpp"
#include "../apps/cli_lookup.hpp"
#include <covoxslam/store/dirty_transfer.hpp>
#include <charconv>
#include <iostream>
#include <iomanip>

namespace covoxslam::metrics
{
struct dirty_arguments
{
    std::filesystem::path input;
    layout selected{layout::pooled_aos};
    std::size_t stride{7};
    bool esdf{};
};
enum class dirty_option { unknown, input, layout, stride, field };
constexpr cli_option_entry<dirty_option> dirty_options[]{
    {"--input", dirty_option::input}, {"--layout", dirty_option::layout},
    {"--stride", dirty_option::stride}, {"--field", dirty_option::field}
};
static_assert(valid_cli_options(dirty_options));
operation_result<dirty_arguments> parse_dirty(int argc, char** argv)
{
    dirty_arguments result;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) return std::unexpected(operation_error{"missing dirty option value"});
        const std::string_view text = argv[i + 1];
        switch (lookup_cli_option(argv[i], dirty_options)) {
        case dirty_option::input: result.input = text; break;
        case dirty_option::layout: {
            auto selected = parse_layout(text);
            if (!selected) return std::unexpected(selected.error());
            result.selected = *selected; break;
        }
        case dirty_option::stride: {
            const auto converted = std::from_chars(text.data(), text.data() + text.size(), result.stride);
            if (converted.ec != std::errc{} || converted.ptr != text.data() + text.size() || !result.stride || result.stride > 1000000)
                return std::unexpected(operation_error{"invalid dirty stride"});
            break;
        }
        case dirty_option::field:
            if (text != "tsdf" && text != "esdf") return std::unexpected(operation_error{"invalid dirty field"});
            result.esdf = text == "esdf"; break;
        default: return std::unexpected(operation_error{"unknown dirty option"});
        }
    }
    if (result.input.empty()) return std::unexpected(operation_error{"dirty input is required"});
    return result;
}
template<schema s, layout l> operation_status measure_dirty(const dirty_arguments& args)
{
    std::vector<store::block_key> keys;
    auto loaded = load_tiles<s>(args.input, 0, 100000, &keys);
    if (!loaded) return std::unexpected(loaded.error());
    const auto start = clock::now();
    draft<s, l> source, readback;
    auto checked = fill_draft(source, std::span<const aos_tile<s>>(*loaded));
    if (!checked) return checked;
    checked = readback.initialize(source.size());
    if (!checked) return checked;
    cuda_draft<s, l> device;
    checked = device.initialize(source.size());
    if (!checked) return checked;
    transfer_measurement initial;
    checked = device.upload_range(source, 0, 0, source.size() * 512, (1U << field_count<s>) - 1, transfer_method::bulk_same_layout, initial);
    if (!checked) return checked;
    store::cpu_block_index index;
    checked = index.initialize(0, s == schema::tsdf ? store::field_kind::tsdf : store::field_kind::esdf, keys);
    if (!checked) return checked;
    store::dirty_tracker<s> dirty(0);
    checked = dirty.reserve(source.size());
    if (!checked) return checked;

    for (std::size_t slot = 0; slot < source.size(); ++slot) {
        auto handle = index.find(keys[slot]);
        if (!handle) return std::unexpected(handle.error());
        checked = dirty.mark(*handle, (1U << field_count<s>) - 1, 1);
        if (!checked) return checked;
    }
    checked = dirty.acknowledge(dirty.capture(), {});
    if (!checked) return checked;
    const auto setup_ms = elapsed(start);
    const auto update = clock::now();

    for (std::size_t slot = 0; slot < source.size(); slot += args.stride) {
        auto handle = index.find(keys[slot]);
        if (!handle) return std::unexpected(handle.error());
        checked = dirty.mark(*handle, 1U, 2);
        if (!checked) return checked;

        for (std::size_t voxel = 0; voxel < 512; ++voxel) source.template scalar<0>(slot, voxel) += .125f;
    }
    const auto update_ms = elapsed(update);
    auto batch = dirty.capture();
    const auto transfer = clock::now();
    auto measured = store::upload_dirty(source, device, index, 2, batch);
    if (!measured) return std::unexpected(measured.error());
    const auto transfer_ms = elapsed(transfer);
    checked = dirty.acknowledge(batch, {});
    if (!checked) return checked;
    transfer_measurement verification;
    const auto verify = clock::now();
    checked = device.download_range(readback, 0, 0, source.size() * 512, (1U << field_count<s>) - 1, transfer_method::bulk_same_layout, verification);
    if (!checked) return checked;
    const auto readback_ms = elapsed(verify);

    for (std::size_t tile = 0; tile < source.size(); ++tile)
        for (std::size_t voxel = 0; voxel < 512; ++voxel) {
            auto expected = gather_record(source.view()[tile].voxels[voxel]);
            auto actual = gather_record(readback.view()[tile].voxels[voxel]);
            if (std::memcmp(&expected, &actual, sizeof(expected))) return std::unexpected(operation_error{"dirty mirror value mismatch"});
        }
    constexpr auto tile_bytes = storage_traits<s, l>::tile_stride * (l == layout::pool_soa ? field_count<s> : 1);
    std::cout << std::setprecision(17) << layout_name(l) << ',' << (s == schema::tsdf ? "tsdf" : "esdf") << ',' << source.size() << ','
        << args.stride << ',' << setup_ms << ',' << update_ms << ',' << transfer_ms << ',' << readback_ms << ','
        << measured->logical_bytes << ',' << measured->transferred_bytes << ',' << measured->calls << ',' << measured->waits << ','
        << measured->ranges << ',' << source.capacity() * tile_bytes << ',' << verification.useful_bytes << ",0\n";
    return {};
}
}
int main(int argc, char** argv)
{
    using namespace covoxslam;
    auto args = metrics::parse_dirty(argc, argv);
    const auto fail = [](const operation_error& error) { std::cerr << error.message << '\n'; return 2; };
    if (!args) return fail(args.error());
    std::cout << "layout,field,tiles,stride,setup_ms,cpu_update_ms,dirty_upload_ms,verification_readback_ms,logical_bytes,transferred_bytes,calls,waits,ranges,payload_capacity_bytes,verification_bytes,mismatches\n";
    auto checked = storage::dispatch_layout(args->selected, [&]<storage::layout l>() {
        return args->esdf ? metrics::measure_dirty<storage::schema::esdf, l>(*args) : metrics::measure_dirty<storage::schema::tsdf, l>(*args);
    });
    return checked ? 0 : fail(checked.error());
}
