#include "nested_workload.hpp"
#include "../apps/cli_lookup.hpp"
#include <charconv>
#include <iomanip>
#include <iostream>

namespace covoxslam::metrics
{
struct nested_arguments
{
    std::filesystem::path input, output;
    nested_options settings;
    nested_array_policy policy{nested_array_policy::soa};
    std::size_t last{}, limit{}, lanes{16};
    bool esdf{}, tagged{true};
};
enum class nested_option { unknown, input, output, last, limit, field, lanes, order, schedule, array, tagged, repeats, warmup, stabilization, device };
constexpr cli_option_entry<nested_option> nested_choices[]{
    {"--input", nested_option::input}, {"--output", nested_option::output}, {"--last-submap", nested_option::last},
    {"--limit-tiles", nested_option::limit}, {"--field", nested_option::field}, {"--lanes", nested_option::lanes},
    {"--order", nested_option::order}, {"--schedule", nested_option::schedule}, {"--array", nested_option::array},
    {"--tagged", nested_option::tagged}, {"--repeats", nested_option::repeats}, {"--warmup", nested_option::warmup},
    {"--stabilization", nested_option::stabilization}, {"--device", nested_option::device}
};
static_assert(valid_cli_options(nested_choices));
operation_result<nested_arguments> parse_nested(int argc, char** argv)
{
    nested_arguments result;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) return std::unexpected(operation_error{"missing nested option value"});
        const auto option = lookup_cli_option(argv[i], nested_choices);
        std::string_view value = argv[i + 1];
        int number{};
        if (option == nested_option::last || option == nested_option::limit || option == nested_option::lanes || option == nested_option::repeats || option == nested_option::warmup) {
            auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || number < 0 || number > 100000)
                return std::unexpected(operation_error{"invalid nested count"});
        }
        switch (option) {
        case nested_option::input: result.input = value; break;
        case nested_option::output: result.output = value; break;
        case nested_option::device:
            if (value != "cpu" && value != "cuda") return std::unexpected(operation_error{"nested device must be cpu or cuda"});
            result.settings.cuda = value == "cuda"; break;
        case nested_option::last: result.last = number; break;
        case nested_option::limit: result.limit = number; break;
        case nested_option::lanes: result.lanes = number; break;
        case nested_option::warmup: result.settings.sampling.warmup = number; break;
        case nested_option::repeats: result.settings.sampling.repeats = number; break;
        case nested_option::field:
            if (value != "tsdf" && value != "esdf") return std::unexpected(operation_error{"field must be tsdf or esdf"});
            result.esdf = value == "esdf"; break;
        case nested_option::tagged:
            if (value != "true" && value != "false") return std::unexpected(operation_error{"tagged must be true or false"});
            result.tagged = value == "true"; break;
        case nested_option::order:
            if (value != "parent" && value != "lane") return std::unexpected(operation_error{"order must be parent or lane"});
            result.settings.order = value == "lane" ? nested_order::lane_major : nested_order::parent_major; break;
        case nested_option::schedule:
            if (value != "canonical" && value != "lane") return std::unexpected(operation_error{"schedule must be canonical or lane"});
            result.settings.lane_schedule = value == "lane"; break;
        case nested_option::array:
            if (value != "aos" && value != "soa" && value != "standard") return std::unexpected(operation_error{"array must be aos, soa or standard"});
            result.policy = value == "aos" ? nested_array_policy::aos : value == "soa" ? nested_array_policy::soa : nested_array_policy::standard; break;
        case nested_option::stabilization:
            if (value != "fixed" && value != "bounded") return std::unexpected(operation_error{"stabilization must be fixed or bounded"});
            result.settings.sampling.stability.enabled = value == "bounded"; break;
        default: return std::unexpected(operation_error{"unknown nested option"});
        }
    }
    if (result.input.empty() || (result.lanes != 8 && result.lanes != 16 && result.lanes != 32) ||
        result.settings.sampling.repeats < 1 || result.settings.sampling.repeats > 1000 || result.settings.sampling.warmup > 1000)
        return std::unexpected(operation_error{"nested input lanes or sampling counts are invalid"});
#if !defined(covoxslam_has_cuda)
    if (result.settings.cuda) return std::unexpected(operation_error{"nested CUDA support was not built"});
#endif
    return result;
}
template<schema s, std::size_t lanes, bool tagged> operation_result<measurement> select_nested(
    std::span<const aos_tile<s>> source, const nested_arguments& args)
{
    switch (args.policy) {
    case nested_array_policy::aos: return measure_nested<s, lanes, nested_array_policy::aos, tagged>(source, args.settings);
    case nested_array_policy::soa: return measure_nested<s, lanes, nested_array_policy::soa, tagged>(source, args.settings);
    case nested_array_policy::standard: return measure_nested<s, lanes, nested_array_policy::standard, tagged>(source, args.settings);
    default: return std::unexpected(operation_error{"unsupported nested policy"});
    }
}
template<schema s> operation_status run_nested(const nested_arguments& args, std::ostream& output)
{
    auto loaded = load_tiles<s>(args.input, args.last);
    if (!loaded) return std::unexpected(loaded.error());
    const std::span<const aos_tile<s>> source(loaded->data(), args.limit ? std::min(args.limit, loaded->size()) : loaded->size());
    const auto execute = [&]<std::size_t lanes>() {
        return args.tagged ? select_nested<s, lanes, true>(source, args) : select_nested<s, lanes, false>(source, args);
    };
    auto result = args.lanes == 8 ? execute.template operator()<8>() : args.lanes == 16 ? execute.template operator()<16>() : execute.template operator()<32>();
    if (!result) return std::unexpected(result.error());
    output << std::setprecision(17) << "device,lanes,array,tagged,order,schedule,tiles,reserved_bytes,sample_ms,setup_ms,input_hash,output_hash,mismatches,stability,upload_ms,download_mean_ms,warmup_samples_ms\n";
    std::string warmup;
    for (double value : result->stability.warmup_ms) warmup += std::to_string(value) + ';';

    for (double sample : result->samples_ms)
        output << (args.settings.cuda ? "cuda," : "cpu,") << args.lanes << ',' << int(args.policy) << ',' << args.tagged << ',' << int(args.settings.order) << ',' << args.settings.lane_schedule << ','
            << source.size() << ',' << result->bytes << ',' << sample << ',' << result->setup_ms << ',' << hash_values<aos_tile<s>>(source) << ','
            << result->output_hash << ',' << result->mismatch_count << ',' << stability_name(result->stability) << ',' << result->upload_ms << ',' << result->download_ms << ',' << warmup << '\n';
    if (!output || result->mismatch_count) return std::unexpected(operation_error{"nested output failed or differs from reference"});
    return {};
}
}
int main(int argc, char** argv)
{
    using namespace covoxslam;
    const auto fail = [](const operation_error& error) { std::cerr << error.message << '\n'; return 2; };
    auto args = metrics::parse_nested(argc, argv);
    if (!args) return fail(args.error());
    std::ofstream file;
    if (!args->output.empty()) { file.open(args->output); if (!file) return fail({"cannot open nested metrics output"}); }
    auto& output = args->output.empty() ? std::cout : file;
    auto result = args->esdf ? metrics::run_nested<storage::schema::esdf>(*args, output) : metrics::run_nested<storage::schema::tsdf>(*args, output);
    output.flush();
    if (!output) return fail({"nested output flush failed"});
    return result ? 0 : fail(result.error());
}
