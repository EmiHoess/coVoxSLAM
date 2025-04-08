#pragma once
#include "layout_cpu.hpp"
#include <covoxslam/storage/transfer_session.hpp>

namespace covoxslam::metrics
{
enum class staging_method { pageable_bulk, pinned_blocking, pinned_async };
struct transfer_sample
{
    double upload_submission_ms{}, upload_completion_ms{}, download_submission_ms{}, download_completion_ms{};
    double total_ms{};
};
struct transfer_result
{
    std::vector<transfer_sample> samples;
    stability_report stability;
    double setup_ms{};
    std::size_t pinned_bytes{}, logical_bytes{}, mismatches{};
    std::uint64_t output_hash{};
};
template<schema s, layout l> operation_result<transfer_result> measure_transfers(std::span<const aos_tile<s>> source,
    staging_method method, const measurement_options& options, std::size_t pinned_budget)
{
    if (options.repeats < 1 || options.repeats > 1000)
        return std::unexpected(operation_error{"invalid transfer repetition count"});
    const auto setup = clock::now();
    draft<s, l> input, output;
    auto checked = fill_draft(input, source);
    if (!checked) return std::unexpected(checked.error());
    checked = output.initialize(source.size());
    if (!checked) return std::unexpected(checked.error());
    cuda_draft<s, l> device;
    transfer_session<s, l> staged;
    checked = method == staging_method::pageable_bulk ? device.initialize(source.size()) : staged.initialize(source.size(), pinned_budget);
    if (!checked) return std::unexpected(checked.error());
    std::vector<value<s>> records(source.size() * 512), reference(records.size());
    record_copy<s>(source, reference);
    transfer_result result;
    result.setup_ms = elapsed(setup);
    result.pinned_bytes = staged.pinned_bytes();
    result.logical_bytes = records.size() * sizeof(value<s>);
    transfer_sample row;
    const auto sample = [&]() -> operation_result<double> {
        row = {};
        const auto begin = clock::now();
        constexpr auto fields = (1U << field_count<s>) - 1;
        if (method == staging_method::pageable_bulk) {
            transfer_measurement measured;
            auto start = clock::now();
            checked = device.upload_range(input, 0, 0, records.size(), fields, transfer_method::bulk_same_layout, measured);
            row.upload_submission_ms = elapsed(start);
            if (!checked) return std::unexpected(checked.error());
            start = clock::now();
            checked = device.download_range(output, 0, 0, records.size(), fields, transfer_method::bulk_same_layout, measured);
            row.download_submission_ms = elapsed(start);
        }
        else if (method == staging_method::pinned_blocking) {
            auto start = clock::now();
            checked = staged.upload_blocking(input);
            row.upload_submission_ms = elapsed(start);
            if (!checked) return std::unexpected(checked.error());
            start = clock::now();
            checked = staged.download_blocking(output);
            row.download_submission_ms = elapsed(start);
        }
        else if (method == staging_method::pinned_async) {
            {
                auto start = clock::now();
                auto ticket = staged.upload(input);
                row.upload_submission_ms = elapsed(start);
                if (!ticket) return std::unexpected(ticket.error());
                start = clock::now();
                checked = ticket->wait();
                row.upload_completion_ms = elapsed(start);
                if (!checked) return std::unexpected(checked.error());
            }
            {
                auto start = clock::now();
                auto ticket = staged.download();
                row.download_submission_ms = elapsed(start);
                if (!ticket) return std::unexpected(ticket.error());
                start = clock::now();
                checked = ticket->copy_to(output);
                row.download_completion_ms = elapsed(start);
            }
        }
        else return std::unexpected(operation_error{"unsupported transfer method"});
        row.total_ms = elapsed(begin);
        if (!checked) return std::unexpected(checked.error());
        record_copy<s>(std::as_const(output).view(), records);
        result.output_hash = hash_values<value<s>>(records);

        for (std::size_t i = 0; i < records.size(); ++i)
            result.mismatches += std::memcmp(&records[i], &reference[i], sizeof(value<s>)) != 0;
        if (result.mismatches) return std::unexpected(operation_error{"transfer round trip differs from source"});
        return row.total_ms;
    };
    auto warmup = collect_warmup(options.stability, options.warmup, sample);
    if (!warmup) return std::unexpected(warmup.error());
    result.stability = std::move(*warmup);

    for (int i = 0; i < options.repeats; ++i) {
        auto measured = sample();
        if (!measured) return std::unexpected(measured.error());
        result.samples.push_back(row);
    }
    return result;
}
}
