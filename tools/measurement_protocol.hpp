#pragma once
#include <covoxslam/result.hpp>
#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace covoxslam::metrics
{
struct stability_options
{
    bool enabled{};
    std::size_t minimum{10}, maximum{100}, window{5};
    double relative_median{0.03}, relative_spread{0.10};
};
struct stability_report
{
    std::vector<double> warmup_ms;
    bool evaluated{}, stable{};
};
inline operation_status validate_stability(const stability_options& value)
{
    if (!value.window || value.window > 500 || value.minimum < 2 * value.window ||
        value.maximum < value.minimum || value.maximum > 1000 ||
        !std::isfinite(value.relative_median) || value.relative_median < 0 || value.relative_median > 1 ||
        !std::isfinite(value.relative_spread) || value.relative_spread < 0 || value.relative_spread > 1)
        return std::unexpected(operation_error{"invalid stabilization thresholds"});
    return {};
}
inline double sample_median(std::span<const double> samples)
{
    if (samples.empty()) return 0;
    std::vector<double> ordered(samples.begin(), samples.end());
    std::sort(ordered.begin(), ordered.end());
    const auto middle = ordered.size() / 2;
    return ordered.size() % 2 ? ordered[middle] : (ordered[middle - 1] + ordered[middle]) * 0.5;
}
inline double sample_spread(std::span<const double> samples)
{
    if (samples.empty()) return 0;
    const auto [minimum, maximum] = std::minmax_element(samples.begin(), samples.end());
    return *maximum - *minimum;
}
inline bool stable_windows(std::span<const double> values, const stability_options& options)
{
    if (values.size() < options.minimum || values.size() < 2 * options.window) return false;
    const auto a = values.subspan(values.size() - 2 * options.window, options.window);
    const auto b = values.last(options.window);
    const auto first = sample_median(a), last = sample_median(b);
    if (first <= 0 || last <= 0) return false;
    return std::abs(first - last) <= options.relative_median * std::max(first, last) &&
        sample_spread(a) <= options.relative_spread * first && sample_spread(b) <= options.relative_spread * last;
}
template<class function> operation_result<stability_report> collect_warmup(
    const stability_options& options, int fixed_count, function&& action)
{
    auto checked = validate_stability(options);
    if (!checked) return std::unexpected(checked.error());
    if (fixed_count < 0 || fixed_count > 1000) return std::unexpected(operation_error{"invalid fixed warmup count"});
    stability_report result;
    result.evaluated = options.enabled;
    const auto limit = options.enabled ? options.maximum : std::size_t(fixed_count);
    result.warmup_ms.reserve(limit);

    for (std::size_t i = 0; i < limit; ++i) {
        auto measured = action();
        if (!measured) return std::unexpected(measured.error());
        if (!std::isfinite(*measured) || *measured < 0)
            return std::unexpected(operation_error{"invalid warmup duration"});
        result.warmup_ms.push_back(*measured);
        if (options.enabled && result.warmup_ms.size() % options.window == 0 && stable_windows(result.warmup_ms, options)) {
            result.stable = true;
            break;
        }
    }
    return result;
}
inline bool timing_tie(std::span<const double> first, std::span<const double> second, double relative = 0.03)
{
    if (first.empty() || second.empty()) return true;
    const auto a = sample_median(first), b = sample_median(second);
    return std::abs(a - b) <= std::max({relative * std::max(a, b), sample_spread(first), sample_spread(second)});
}
inline const char* stability_name(const stability_report& report) noexcept
{
    if (!report.evaluated) return "not_evaluated";
    return report.stable ? "stable" : "not_stable";
}
inline operation_status validate_samples(std::span<const double> samples)
{
    if (samples.empty()) return std::unexpected(operation_error{"no measured samples"});

    for (double sample : samples)
        if (!std::isfinite(sample) || sample < 0)
            return std::unexpected(operation_error{"invalid measured duration"});
    return {};
}
}
