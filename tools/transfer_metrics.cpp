#include "transfer_workload.hpp"
#include "../apps/cli_lookup.hpp"
#include <charconv>
#include <iomanip>
#include <iostream>

namespace covoxslam::metrics
{
struct transfer_arguments
{
    std::filesystem::path input, output;
    std::size_t last{}, limit{}, budget{256U * 1024U * 1024U};
    int rounds{4};
    bool esdf{};
    measurement_options sampling;
};
enum class transfer_option { unknown, input, output, last, limit, budget, rounds, warmup, repeats, field, stabilization };
constexpr cli_option_entry<transfer_option> transfer_choices[]{
    {"--input", transfer_option::input}, {"--output", transfer_option::output}, {"--last-submap", transfer_option::last},
    {"--limit-tiles", transfer_option::limit}, {"--pinned-memory-mib", transfer_option::budget}, {"--rounds", transfer_option::rounds},
    {"--warmup", transfer_option::warmup}, {"--repeats", transfer_option::repeats}, {"--field", transfer_option::field},
    {"--stabilization", transfer_option::stabilization}
};
static_assert(valid_cli_options(transfer_choices));
operation_result<transfer_arguments> parse_transfers(int argc, char** argv)
{
    transfer_arguments result;

    for (int i = 1; i < argc; i += 2) {
        if (i + 1 == argc) return std::unexpected(operation_error{"missing transfer option value"});
        const auto option = lookup_cli_option(argv[i], transfer_choices);
        const std::string_view value = argv[i + 1];
        int number{};
        if (option == transfer_option::last || option == transfer_option::limit || option == transfer_option::budget ||
            option == transfer_option::rounds || option == transfer_option::warmup || option == transfer_option::repeats) {
            auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || number < 0 || number > 100000)
                return std::unexpected(operation_error{"invalid transfer count"});
        }
        switch (option) {
        case transfer_option::input: result.input = value; break;
        case transfer_option::output: result.output = value; break;
        case transfer_option::last: result.last = number; break;
        case transfer_option::limit: result.limit = number; break;
        case transfer_option::budget: result.budget = std::size_t(number) * 1024 * 1024; break;
        case transfer_option::rounds: result.rounds = number; break;
        case transfer_option::warmup: result.sampling.warmup = number; break;
        case transfer_option::repeats: result.sampling.repeats = number; break;
        case transfer_option::field:
            if (value != "tsdf" && value != "esdf") return std::unexpected(operation_error{"field must be tsdf or esdf"});
            result.esdf = value == "esdf"; break;
        case transfer_option::stabilization:
            if (value != "fixed" && value != "bounded") return std::unexpected(operation_error{"stabilization must be fixed or bounded"});
            result.sampling.stability.enabled = value == "bounded"; break;
        default: return std::unexpected(operation_error{"unknown transfer option"});
        }
    }
    if (result.input.empty() || result.rounds < 1 || result.rounds > 1000 || result.sampling.repeats < 1 ||
        result.sampling.repeats > 1000 || result.sampling.warmup > 1000 || result.budget > std::size_t(16384) * 1024 * 1024)
        return std::unexpected(operation_error{"transfer input or sampling settings are invalid"});
    return result;
}
template<schema s> operation_status run_transfers(const transfer_arguments& args, std::ostream& output)
{
    auto loaded = load_tiles<s>(args.input, args.last);
    if (!loaded) return std::unexpected(loaded.error());
    const std::span<const aos_tile<s>> source(loaded->data(), args.limit ? std::min(args.limit, loaded->size()) : loaded->size());
    const auto input_hash = hash_values<aos_tile<s>>(source);
    const char* methods[]{"pageable_bulk", "pinned_blocking", "pinned_async"};
    output << std::setprecision(17) << "round,layout,method,tiles,logical_bytes,pinned_bytes,setup_ms,upload_call_ms,upload_wait_ms,download_call_ms,download_wait_and_unpack_ms,total_ms,input_hash,output_hash,mismatches,stability,warmup_total_ms\n";

    for (int round = 0; round < args.rounds; ++round)
        for (int order = 0; order < 9; ++order) {
            const auto choice = (round + order) % 9;
            const auto method = staging_method(choice / 3);
            const auto selected = layout(choice % 3);
            if (!layout_available(selected)) continue;
            auto checked = dispatch_layout(selected, [&]<layout l>() -> operation_status {
                auto measured = measure_transfers<s, l>(source, method, args.sampling, args.budget);
                if (!measured) return std::unexpected(measured.error());
                std::string warmup;
                for (double value : measured->stability.warmup_ms) warmup += std::to_string(value) + ';';

                for (const auto& row : measured->samples)
                    output << round << ',' << layout_name(selected) << ',' << methods[int(method)] << ',' << source.size() << ',' << measured->logical_bytes << ','
                        << measured->pinned_bytes << ',' << measured->setup_ms << ',' << row.upload_submission_ms << ',' << row.upload_completion_ms << ','
                        << row.download_submission_ms << ',' << row.download_completion_ms << ',' << row.total_ms << ',' << input_hash << ','
                        << measured->output_hash << ',' << measured->mismatches << ',' << stability_name(measured->stability) << ',' << warmup << '\n';
                if (!output) return std::unexpected(operation_error{"cannot write transfer measurements"});
                return {};
            });
            if (!checked) return checked;
        }
    return {};
}
}
int main(int argc, char** argv)
{
    using namespace covoxslam;
    const auto fail = [](const operation_error& error) { std::cerr << error.message << '\n'; return 2; };
    auto args = metrics::parse_transfers(argc, argv);
    if (!args) return fail(args.error());
    std::ofstream file;
    if (!args->output.empty()) { file.open(args->output); if (!file) return fail({"cannot open transfer metrics output"}); }
    auto& output = args->output.empty() ? std::cout : file;
    auto result = args->esdf ? metrics::run_transfers<storage::schema::esdf>(*args, output) : metrics::run_transfers<storage::schema::tsdf>(*args, output);
    output.flush();
    if (!output) return fail({"transfer output flush failed"});
    return result ? 0 : fail(result.error());
}
