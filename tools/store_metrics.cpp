#include "layout_workload.hpp"
#include "../apps/cli_lookup.hpp"
#include <covoxslam/store/store.hpp>
#include <charconv>
#include <iostream>

namespace covoxslam::metrics
{
struct store_arguments {
    std::filesystem::path input;
    std::size_t last{};
    std::size_t snapshot_budget{512U * 1024U * 1024U};
    layout selected{layout::pooled_aos};
    bool persistent{true};
};
enum class store_option { unknown, input, last, layout, store, budget };
constexpr cli_option_entry<store_option> store_options[]{
    {"--input", store_option::input}, {"--last-submap", store_option::last},
    {"--layout", store_option::layout}, {"--store", store_option::store}, {"--snapshot-memory-mib", store_option::budget}
};
static_assert(valid_cli_options(store_options));
operation_result<store_arguments> parse_store(int argc, char** argv)
{
    store_arguments result;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) return std::unexpected(operation_error{"missing store option value"});
        const std::string_view value = argv[i + 1];
        switch (lookup_cli_option(argv[i], store_options)) {
        case store_option::input: result.input = value; break;
        case store_option::budget: {
            std::size_t mib{};
            const auto converted = std::from_chars(value.data(), value.data() + value.size(), mib);
            if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || mib > 16384)
                return std::unexpected(operation_error{"invalid snapshot budget"});
            result.snapshot_budget = mib * 1024 * 1024; break;
        }
        case store_option::last: {
            const auto converted = std::from_chars(value.data(), value.data() + value.size(), result.last);
            if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || result.last > 1000)
                return std::unexpected(operation_error{"invalid submap limit"});
            break;
        }
        case store_option::layout: {
            auto selected = parse_layout(value);
            if (!selected) return std::unexpected(selected.error());
            result.selected = *selected; break;
        }
        case store_option::store:
            if (value != "persistent" && value != "legacy") return std::unexpected(operation_error{"invalid store selection"});
            result.persistent = value == "persistent"; break;
        default: return std::unexpected(operation_error{"unknown store option"});
        }
    }
    if (result.input.empty()) return std::unexpected(operation_error{"store input is required"});
    return result;
}
template<schema s, layout l> operation_status import_maps(store::map_store<l>& maps, const store_arguments& args)
{
    std::vector<store::block_key> keys;
    auto source = load_tiles<s>(args.input, args.last, 100000, &keys);
    if (!source) return std::unexpected(source.error());
    std::vector<value<s>> output, expected;
    std::size_t first{};
    double import_ms{}, read_ms{};

    for (std::uint64_t id = 0; id <= args.last; ++id) {
        std::size_t last = first;
        while (last < keys.size() && keys[last].submap == id) ++last;
        std::vector<value<s>> records;

        for (auto tile = first; tile < last; ++tile)
            records.insert(records.end(), (*source)[tile].voxels.begin(), (*source)[tile].voxels.end());
        expected.insert(expected.end(), records.begin(), records.end());
        const auto started = clock::now();
        if (args.persistent) {
            auto checked = maps.template import_field<s>(id, std::span(keys).subspan(first, last - first), records);
            if (!checked) return checked;
        }
        import_ms += elapsed(started);
        const auto read_start = clock::now();
        if (args.persistent) {
            const std::array<std::uint64_t, 1> ids{id};
            auto snapshot = maps.snapshot(ids, s == schema::tsdf ? 1U : 2U, args.snapshot_budget);
            if (!snapshot) return std::unexpected(snapshot.error());
            auto field = [&]() { if constexpr (s == schema::tsdf) return snapshot->front().tsdf; else return snapshot->front().esdf; }();
            auto view = field->payload.view();

            for (std::size_t tile = 0; tile < view.size(); ++tile)
                for (std::size_t voxel = 0; voxel < 512; ++voxel)
                    output.push_back(gather_record(view[tile].voxels[voxel]));
        }
        else output.insert(output.end(), records.begin(), records.end());
        read_ms += elapsed(read_start);
        first = last;
    }
    if (expected.size() != output.size() || std::memcmp(expected.data(), output.data(), expected.size() * sizeof(value<s>)))
        return std::unexpected(operation_error{"store canonical payload differs"});
    std::cout << (s == schema::tsdf ? "tsdf" : "esdf") << ',' << (args.persistent ? "persistent" : "legacy") << ','
        << layout_name(l) << ',' << maps.size() << ',' << source->size() << ',' << import_ms << ',' << read_ms << ','
        << hash_values<value<s>>(expected) << ',' << hash_values<value<s>>(output) << '\n';
    return {};
}
}
int main(int argc, char** argv)
{
    using namespace covoxslam;
    auto args = metrics::parse_store(argc, argv);
    const auto fail = [](const operation_error& error) { std::cerr << error.message << '\n'; return 2; };
    if (!args) return fail(args.error());
    std::cout << "field,store,layout,submaps,tiles,import_ms,read_ms,input_hash,output_hash\n";
    auto checked = storage::dispatch_layout(args->selected, [&]<storage::layout l>() -> operation_status {
        store::map_store<l> maps;

        for (std::uint64_t id = 0; id <= args->last; ++id) {
            auto created = maps.create(id);
            if (!created) return created;
        }
        auto tsdf = metrics::import_maps<storage::schema::tsdf>(maps, *args);
        if (!tsdf) return tsdf;
        return metrics::import_maps<storage::schema::esdf>(maps, *args);
    });
    return checked ? 0 : fail(checked.error());
}
