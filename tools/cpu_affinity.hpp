#pragma once
#include <covoxslam/result.hpp>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

namespace covoxslam::metrics
{
class cpu_affinity
{
    bool changed_{};
#if defined(_WIN32)
    DWORD_PTR previous_{};
#elif defined(__linux__)
    cpu_set_t previous_{};
#endif
public:
    cpu_affinity() = default;
    cpu_affinity(const cpu_affinity&) = delete;
    cpu_affinity& operator=(const cpu_affinity&) = delete;
    ~cpu_affinity()
    {
#if defined(_WIN32)
        if (changed_) SetThreadAffinityMask(GetCurrentThread(), previous_);
#elif defined(__linux__)
        if (changed_) sched_setaffinity(0, sizeof(previous_), &previous_);
#endif
    }
    operation_status select(int cpu)
    {
        if (changed_) return std::unexpected(operation_error{"CPU affinity was already selected"});
        if (cpu == -1) return {};
        if (cpu < 0 || cpu >= 64) return std::unexpected(operation_error{"CPU index must be 0 through 63"});
#if defined(_WIN32)
        DWORD_PTR process{}, system{};
        const DWORD_PTR mask = DWORD_PTR{1} << cpu;
        if (!GetProcessAffinityMask(GetCurrentProcess(), &process, &system) || !(process & mask))
            return std::unexpected(operation_error{"CPU is unavailable in the current processor group"});
        previous_ = SetThreadAffinityMask(GetCurrentThread(), mask);
        if (!previous_) return std::unexpected(operation_error{"cannot set CPU affinity"});
#elif defined(__linux__)
        if (sched_getaffinity(0, sizeof(previous_), &previous_) != 0 || !CPU_ISSET(cpu, &previous_))
            return std::unexpected(operation_error{"CPU is unavailable in the current affinity set"});
        cpu_set_t selected;
        CPU_ZERO(&selected);
        CPU_SET(cpu, &selected);
        if (sched_setaffinity(0, sizeof(selected), &selected) != 0)
            return std::unexpected(operation_error{"cannot set CPU affinity"});
#else
        return std::unexpected(operation_error{"explicit CPU affinity is unsupported on this platform"});
#endif
        changed_ = true;
        return {};
    }
};
}
