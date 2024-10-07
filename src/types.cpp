#include <covoxslam/types.hpp>

#include <cmath>
#include <numbers>

namespace covoxslam {

float length(vec3f value) noexcept {
    return std::sqrt(dot(value, value));
}

vec3f normalized(vec3f value) noexcept {
     const float magnitude = length(value);
    return magnitude > 1.0e-20F && std::isfinite(magnitude)
        ? value / magnitude : vec3f{};
}

quaternion normalized(quaternion value) noexcept {
    const float magnitude = std::sqrt(value.x * value.x + value.y * value.y +
                                      value.z * value.z + value.w * value.w);
    if (!(magnitude > 0.0F) || !std::isfinite(magnitude)) return {};
    return {value.x / magnitude, value.y / magnitude,
             value.z / magnitude, value.w / magnitude};
}

quaternion multiply(quaternion a, quaternion b) noexcept {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

quaternion quaternion_from_yaw(double yaw) noexcept {
   return {0.0F, 0.0F, static_cast<float>(std::sin(0.5 * yaw)),
            static_cast<float>(std::cos(0.5 * yaw))};
}

vec3f rotate_vector(quaternion rotation, vec3f value) noexcept {
    const quaternion unit = normalized(rotation);
    const vec3f q{unit.x, unit.y, unit.z};
    const vec3f twice_cross = cross(q, value) * 2.0F;
    return value + twice_cross * unit.w + cross(q, twice_cross);
}

vec3f transform_point(const pose& transform, vec3f point) noexcept {
     return transform.translation + rotate_vector(transform.rotation, point);
}

vec3f inverse_transform_point(const pose& transform, vec3f point) noexcept {
    const quaternion inverse{-transform.rotation.x, -transform.rotation.y,
                             -transform.rotation.z, transform.rotation.w};
    return rotate_vector(inverse, point - transform.translation);
}

double yaw_of(quaternion rotation) noexcept {
    const quaternion q = normalized(rotation);
    return std::atan2(2.0 * (static_cast<double>(q.w) * q.z +
                              static_cast<double>(q.x) * q.y),
                      1.0 - 2.0 * (static_cast<double>(q.y) * q.y +
                                   static_cast<double>(q.z) * q.z));
}

double wrap_yaw(double yaw) noexcept {
    if (!std::isfinite(yaw)) return yaw;
    const double pi = std::numbers::pi;
    double result = std::fmod(yaw + pi, 2.0 * pi);
    if (result < 0.0) result += 2.0 * pi;
     return result - pi;
}

}
