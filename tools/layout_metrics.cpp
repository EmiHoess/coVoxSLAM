#include "layout_cpu.hpp"
#if defined(covoxslam_has_cuda)
#include "layout_gpu.hpp"
#endif
#include "../apps/cli_lookup.hpp"
#include <charconv>
#include <iomanip>
#include <iostream>

namespace covoxslam::metrics
{
struct arguments
{
    std::filesystem::path input, output;
    std::size_t last_submap{};
    std::size_t limit_tiles{};
    measurement_options sampling;
    int rounds{4};
    bool esdf{}, cpu{true}, cuda{};
};
enum class option { unknown, input, output, last, warmup, repeats, rounds, field, device, copy, batch, packing, query, stride, limit };
constexpr cli_option_entry<option> choices[]{
    {"--input", option::input}, {"--output", option::output}, {"--last-submap", option::last},
    {"--warmup", option::warmup}, {"--repeats", option::repeats}, {"--rounds", option::rounds},
    {"--field", option::field}, {"--device", option::device}, {"--copy", option::copy},
    {"--packing-batch", option::batch}, {"--packing-policy", option::packing},
    {"--query", option::query}, {"--query-stride", option::stride}, {"--limit-tiles", option::limit}
};
static_assert(valid_cli_options(choices));
enum class copy_choice { unknown, scalar, range, packed, unpacked };
constexpr cli_option_entry<copy_choice> copy_choices[]{
    {"scalar", copy_choice::scalar}, {"range", copy_choice::range},
    {"packed", copy_choice::packed}, {"unpacked", copy_choice::unpacked}
};
static_assert(valid_cli_options(copy_choices));
enum class query_choice { unknown, scalar, tile, contiguous, indexed };
constexpr cli_option_entry<query_choice> query_choices[]{
    {"scalar", query_choice::scalar}, {"tile", query_choice::tile},
    {"contiguous", query_choice::contiguous}, {"indexed", query_choice::indexed}
};
static_assert(valid_cli_options(query_choices));
operation_result<arguments> parse_arguments(int argc, char** argv)
{
    arguments result;

    for (int index = 1; index < argc; index += 2) {
        if (index + 1 == argc) return std::unexpected(operation_error{"missing metrics option value"});
        const std::string_view value = argv[index + 1];
        const auto selected = lookup_cli_option(argv[index], choices);
        int number{};
        if (selected == option::last || selected == option::warmup || selected == option::repeats || selected == option::rounds || selected == option::batch || selected == option::stride || selected == option::limit) {
            const auto converted = std::from_chars(value.data(), value.data() + value.size(), number);
            if (converted.ec != std::errc{} || converted.ptr != value.data() + value.size() || number < 0 ||
                number > (selected == option::batch ? 4096 : selected == option::limit || selected == option::stride ? 100000 : 1000))
                return std::unexpected(operation_error{"metrics count is outside supported limits"});
        }
        switch (selected)
        {
        case option::input: result.input = value; break;
        case option::output: result.output = value; break;
        case option::last: result.last_submap = std::size_t(number); break;
        case option::warmup: result.sampling.warmup = number; break;
        case option::repeats: result.sampling.repeats = number; break;
        case option::rounds: result.rounds = number; break;
        case option::stride: result.sampling.query_stride = std::size_t(number); break;
        case option::limit: result.limit_tiles = std::size_t(number); break;
        case option::query:
            switch (lookup_cli_option(value, query_choices)) {
            case query_choice::scalar: result.sampling.query = query_method::scalar; break;
            case query_choice::tile: result.sampling.query = query_method::tile; break;
            case query_choice::contiguous: result.sampling.query = query_method::contiguous; break;
            case query_choice::indexed: result.sampling.query = query_method::indexed; break;
            default: return std::unexpected(operation_error{"query must be scalar, tile, contiguous or indexed"});
            }
            break;
        case option::copy:
            switch (lookup_cli_option(value, copy_choices)) {
            case copy_choice::scalar: result.sampling.method = copy_method::scalar; break;
            case copy_choice::range: result.sampling.method = copy_method::range; break;
            case copy_choice::packed: result.sampling.method = copy_method::packed; break;
            case copy_choice::unpacked: result.sampling.method = copy_method::unpacked; break;
            default: return std::unexpected(operation_error{"copy must be scalar, range, packed or unpacked"});
            }
            break;
        case option::batch: result.sampling.batch_records = std::size_t(number); break;
        case option::packing:
            if (value != "portable" && value != "fixed") return std::unexpected(operation_error{"packing policy must be portable or fixed"});
            result.sampling.fixed_batch = value == "fixed"; break;
        case option::field:
            if (value != "tsdf" && value != "esdf") return std::unexpected(operation_error{"field must be tsdf or esdf"});
            result.esdf = value == "esdf"; break;
        case option::device:
            if (value != "cpu" && value != "cuda" && value != "both") return std::unexpected(operation_error{"device must be cpu, cuda or both"});
            result.cpu = value != "cuda"; result.cuda = value != "cpu"; break;
        default: return std::unexpected(operation_error{"unknown metrics option"});
        }
    }
    if (result.input.empty() || result.rounds == 0 || result.sampling.repeats == 0 || result.sampling.query_stride == 0)
        return std::unexpected(operation_error{"input and positive rounds and repeats are required"});
    auto packing = select_packing_batch(result.sampling, [](auto value) { return validate_packing(value); });
    if (!packing) return std::unexpected(packing.error());
    if (result.sampling.method != copy_method::scalar && result.cuda)
        return std::unexpected(operation_error{"range copy measurements require the CPU device"});
    if (result.sampling.query != query_method::scalar && result.cuda)
        return std::unexpected(operation_error{"field stream measurements require the CPU device"});
#if !defined(covoxslam_has_cuda)
    if (result.cuda) return std::unexpected(operation_error{"CUDA metrics were not built"});
#endif
    return result;
}
template<schema s, bool copy> operation_result<measurement> measure_selected(std::span<const aos_tile<s>> source,
    const arguments& settings, layout selected, bool direct, bool gpu)
{
    if (direct && !gpu) return measure_cpu<s, copy>(source, source, settings.sampling);
    operation_result<measurement> result = std::unexpected(operation_error{"layout not selected"});
    auto dispatched = dispatch_layout(selected, [&]<layout l>() -> operation_status {
#if defined(covoxslam_has_cuda)
        if (gpu) result = measure_gpu<s, l, copy>(source, settings.sampling, direct);
        else
#endif
            result = measure_host_layout<s, l, copy>(source, settings.sampling);
        return result ? operation_status{} : std::unexpected(result.error());
    });
    if (!dispatched) return std::unexpected(dispatched.error());
    return result;
}
template<schema s> operation_status run(const arguments& settings, std::ostream& output)
{
    auto loaded = load_tiles<s>(settings.input, settings.last_submap);
    if (!loaded) return std::unexpected(loaded.error());
    const std::span<const aos_tile<s>> source(loaded->data(), settings.limit_tiles ? std::min(settings.limit_tiles, loaded->size()) : loaded->size());
    const auto input_hash = hash_values<aos_tile<s>>(source);
    output << std::setprecision(17);
    output << "round,device,layout,workload,sample_ms,setup_ms,upload_ms,download_mean_ms,bytes,copy_calls,mismatches,input_hash,output_hash,tiles,packing_batch,fixed_batch,query_method,query_stride\n";

    for (int round = 0; round < settings.rounds; ++round)
        for (int order = 0; order < 4; ++order)
        {
            const int choice = (round + order) % 4;
            const bool direct = choice == 0;
            const auto selected = choice < 2 ? layout::pooled_aos : choice == 2 ? layout::tile_aosoa : layout::pool_soa;
            if (!layout_available(selected)) continue;

            for (bool copy : {false, true})
                for (bool gpu : {false, true}) {
                    if ((gpu && !settings.cuda) || (!gpu && !settings.cpu)) continue;
                    auto result = copy ? measure_selected<s, true>(source, settings, selected, direct, gpu)
                                       : measure_selected<s, false>(source, settings, selected, direct, gpu);
                    if (!result) return std::unexpected(result.error());
                    const auto workload = !copy ? (settings.sampling.query == query_method::indexed ? "indexed_distance_query" : "field_query") : settings.sampling.method == copy_method::unpacked ? "aos_to_layout_unpack" :
                        settings.sampling.method == copy_method::packed ? "layout_to_aos_pack" : settings.sampling.method == copy_method::range ?
                        (choice < 2 ? "aos_range_copy" : "layout_range_to_aos_conversion") :
                        (choice < 2 ? "aos_record_copy" : "layout_to_aos_conversion");

                    for (double sample : result->samples_ms)
                        output << round << ',' << (gpu ? "cuda" : "cpu") << ',' << (direct ? "direct-aos" : layout_name(selected)) << ','
                            << workload << ',' << sample << ',' << result->setup_ms << ',' << result->upload_ms << ',' << result->download_ms << ','
                            << result->bytes << ',' << result->copy_calls << ',' << result->mismatch_count << ',' << input_hash << ',' << result->output_hash << ','
                            << source.size() << ',' << settings.sampling.batch_records << ',' << settings.sampling.fixed_batch << ','
                            << int(settings.sampling.query) << ',' << settings.sampling.query_stride << '\n';
                    if (!output) return std::unexpected(operation_error{"metrics output failed"});
                    if (result->mismatch_count) return std::unexpected(operation_error{"layout workload output mismatch"});
                }
        }
    return {};
}
}
int main(int argc, char** argv)
{
    using namespace covoxslam;
    const auto fail = [](const operation_error& error) { std::cerr << error.message << '\n'; return 2; };
    auto args = metrics::parse_arguments(argc, argv);
    if (!args) return fail(args.error());
    std::ofstream file;
    if (!args->output.empty()) {
        file.open(args->output);
        if (!file) return fail({"cannot open metrics output"});
    }
    auto& output = args->output.empty() ? std::cout : file;
    auto result = args->esdf ? metrics::run<storage::schema::esdf>(*args, output) : metrics::run<storage::schema::tsdf>(*args, output);
    output.flush();
    if (!output) return fail({"cannot flush metrics output"});
    return result ? 0 : fail(result.error());
}
