#pragma once
#include <cstdint>

namespace covoxslam
{
struct input_timing
{
    double read_ms{}, validation_ms{}, decode_ms{}, deskew_ms{}, integration_ms{}, finalization_ms{};
    std::uint64_t payload_bytes{};
    input_timing& operator+=(const input_timing& other) noexcept {
        read_ms += other.read_ms; validation_ms += other.validation_ms;
        decode_ms += other.decode_ms; deskew_ms += other.deskew_ms;
        integration_ms += other.integration_ms; finalization_ms += other.finalization_ms;
        payload_bytes += other.payload_bytes;
        return *this;
    }
    double buffer_to_map_ms() const noexcept
    {
        return decode_ms + deskew_ms + integration_ms;
    }
};
}
