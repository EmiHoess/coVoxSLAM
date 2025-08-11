#pragma once
#include <covoxslam/core/result.hpp>
#include <covoxslam/pgo_publication.hpp>
#include "cli_lookup.hpp"
#include <sstream>

namespace covoxslam
{
struct pgo_cli_options
{
    pgo::options solver;
    bool shadow{};
     bool publication_selected{};
};
enum class pgo_option { unknown, solver, publication };
inline constexpr cli_option_entry<pgo_option> pgo_options[]{
    {"--pgo", pgo_option::solver}, {"--pgo-publication", pgo_option::publication}
};
enum class pgo_value { unknown, none, cpu, cuda, apply, shadow };
inline constexpr cli_option_entry<pgo_value> pgo_values[]{
    {"none", pgo_value::none}, {"cpu", pgo_value::cpu}, {"cuda", pgo_value::cuda},
    {"apply", pgo_value::apply}, {"shadow", pgo_value::shadow}
};
inline operation_result<bool> parse_pgo_option(std::string_view key, std::string_view value, pgo_cli_options& result)
{
    switch (lookup_cli_option(key, pgo_options)) {
    case pgo_option::solver:
        switch (lookup_cli_option(value, pgo_values)) {
        case pgo_value::none: result.solver.enabled = false; break;
        case pgo_value::cpu: result.solver.enabled = true; result.solver.solver = pgo::backend::native_cpu; break;
        case pgo_value::cuda: result.solver.enabled = true; result.solver.solver = pgo::backend::cuda_cold; break;
        default: return std::unexpected(operation_error{"PGO must be none, cpu or cuda"});
        }
        return true;
    case pgo_option::publication:
        switch (lookup_cli_option(value, pgo_values)) {
        case pgo_value::apply: result.shadow = false; break;
        case pgo_value::shadow: result.shadow = true; break;
        default: return std::unexpected(operation_error{"PGO publication must be apply or shadow"});
        }
        result.publication_selected = true;
        return true;
    default: return false;
    }
}
struct pgo_cli_result
{
    std::string backend{"none"}, hash, publication{"none"}, report;
};
inline operation_result<pgo_cli_result> finish_pgo(std::span<submap> maps, const pgo_cli_options& settings)
{
    if (!settings.solver.enabled) return pgo_cli_result{};
#if defined(covoxslam_has_pgo)
    pgo::optimization_options options;
    options.execution = settings.solver;
    auto optimized = pgo::optimize_submaps(maps, options);

    if (!optimized) return std::unexpected(optimized.error());
    if (!*optimized) return std::unexpected(operation_error{"enabled PGO returned no solve"});
    auto& result = **optimized;
    pgo_cli_result output;
    output.backend = ceronoceres::backend_name(result.solution.backend);
    output.hash = result.snapshot.hash();
    output.publication = "rejected";
    std::ostringstream report;
    auto written = pgo::write_solver_report(report, result.solution);
    if (!written) return std::unexpected(written.error());
    written = pgo::write_iteration_report(report, result.solution);
    if (!written) return std::unexpected(written.error());

    if (result.solution.solution_usable) {
        auto update = pgo::prepare_publication(maps, result);
        if (!update) return std::unexpected(update.error());
        const auto mode = settings.shadow ? pgo::publication_mode::shadow : pgo::publication_mode::apply;
        auto applied = pgo::publish_poses(maps, *update, mode);

        if (!applied) return std::unexpected(applied.error());
        output.publication = settings.shadow ? "shadow" : "applied";
        report << "candidate_poses\n";

        for (const auto& pose : update->poses())
            report << pose.x << ' ' << pose.y << ' ' << pose.z << ' ' << pose.yaw << '\n';
    }
    output.report = std::move(report).str();
    return output;
#else
    return std::unexpected(operation_error{"PGO support was not built"});
#endif
}
}
