#include <covoxslam/core/geometry.hpp>

#include <cmath>
#include <numbers>

namespace covoxslam {

float length(vec3f value) noexcept {
    return std::hypot(value.x, value.y, value.z);
}

vec3f normalized(vec3f value) noexcept {
     const double magnitude = std::hypot(static_cast<double>(value.x),
                                         static_cast<double>(value.y),
                                         static_cast<double>(value.z));
    if (!(magnitude > 1.0e-20) || !std::isfinite(magnitude)) return {};
    return {static_cast<float>(value.x / magnitude),
            static_cast<float>(value.y / magnitude),
            static_cast<float>(value.z / magnitude)};
}

quaternion normalized(quaternion value) noexcept {
    const double magnitude = std::hypot(
        std::hypot(static_cast<double>(value.x), static_cast<double>(value.y)),
         std::hypot(static_cast<double>(value.z), static_cast<double>(value.w)));
    if (!(magnitude > 0.0) || !std::isfinite(magnitude)) return {};
    return {static_cast<float>(value.x / magnitude),
            static_cast<float>(value.y / magnitude),
            static_cast<float>(value.z / magnitude),
            static_cast<float>(value.w / magnitude)};
}

quaternion multiply(quaternion a, quaternion b) noexcept {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

quaternion quaternion_from_yaw(double yaw) noexcept {
    if (!std::isfinite(yaw)) return {};
    yaw = wrap_yaw(yaw);
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
    return wrap_yaw(std::atan2(2.0 * (static_cast<double>(q.w) * q.z +
                              static_cast<double>(q.x) * q.y),
                      1.0 - 2.0 * (static_cast<double>(q.y) * q.y +
                                   static_cast<double>(q.z) * q.z)));
}

double wrap_yaw(double yaw) noexcept {
    if (!std::isfinite(yaw)) return yaw;
    const double pi = std::numbers::pi;
    double result = std::fmod(yaw, 2.0 * pi);
    if (result >= pi) result -= 2.0 * pi;
    if (result < -pi) result += 2.0 * pi;
     return result == 0.0 ? 0.0 : result;
}

pose4 to_pose4(const pose& value) noexcept {
    return {value.translation.x, value.translation.y, value.translation.z,
            yaw_of(value.rotation)};
}

pose to_pose(const pose4& value) noexcept {
    return {{static_cast<float>(value.x), static_cast<float>(value.y),
             static_cast<float>(value.z)}, quaternion_from_yaw(value.yaw)};
}

vec3f world_to_local(const pose4& anchor, vec3f world) noexcept {
     const double dx = static_cast<double>(world.x) - anchor.x;
    const double dy = static_cast<double>(world.y) - anchor.y;
    const double cosine = std::cos(wrap_yaw(anchor.yaw));
    const double sine = std::sin(wrap_yaw(anchor.yaw));
    return {static_cast<float>(cosine * dx + sine * dy),
            static_cast<float>(-sine * dx + cosine * dy),
            static_cast<float>(static_cast<double>(world.z) - anchor.z)};
}

pose local_pose(const pose4& anchor, const pose& world_pose) noexcept {
    return {world_to_local(anchor, world_pose.translation),
              normalized(multiply(quaternion_from_yaw(-anchor.yaw),
                                  normalized(world_pose.rotation)))};
}

pose4 relative_pose(const pose4& left, const pose4& right) noexcept {
    const double dx = right.x - left.x;
    const double dy = right.y - left.y;
    const double cosine = std::cos(wrap_yaw(left.yaw));
    const double sine = std::sin(wrap_yaw(left.yaw));
    return {cosine * dx + sine * dy, -sine * dx + cosine * dy,
             right.z - left.z, wrap_yaw(wrap_yaw(right.yaw) - wrap_yaw(left.yaw))};
}

}
