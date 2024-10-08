#pragma once

namespace covoxslam {

struct vec3f {
    float x{};
    float y{};
    float z{};
};

 struct int3 {
    int x{};
    int y{};
    int z{};
    bool operator==(const int3&) const noexcept = default;
};

struct quaternion {
    float x{};
    float y{};
     float z{};
    float w{1.0F};
};

struct pose {
    vec3f translation{};
    quaternion rotation{};
};

struct pose4 {
    double x{};
    double  y{};
    double z{};
    double yaw{};
};

constexpr vec3f operator+(vec3f a, vec3f b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

constexpr vec3f operator-(vec3f a, vec3f b) noexcept {
   return {a.x - b.x, a.y - b.y, a.z - b.z};
}

constexpr vec3f operator*(vec3f value, float scale) noexcept {
    return {value.x * scale, value.y * scale, value.z * scale};
}

constexpr vec3f operator/(vec3f value, float scale) noexcept {
    return {value.x / scale, value.y / scale, value.z / scale};
}

constexpr float dot(vec3f a, vec3f b) noexcept {
     return a.x * b.x + a.y * b.y + a.z * b.z;
}

constexpr vec3f cross(vec3f a, vec3f b) noexcept {
    return {a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

float length(vec3f value) noexcept;
vec3f normalized(vec3f value) noexcept;
 quaternion normalized(quaternion value) noexcept;
quaternion multiply(quaternion a, quaternion b) noexcept;
quaternion quaternion_from_yaw(double yaw) noexcept;
vec3f rotate_vector(quaternion rotation, vec3f value) noexcept;
vec3f transform_point(const pose& transform, vec3f point) noexcept;
vec3f inverse_transform_point(const pose& transform, vec3f point) noexcept;
double yaw_of(quaternion rotation) noexcept;
double wrap_yaw(double yaw) noexcept;
pose4 to_pose4(const pose& value) noexcept;
pose to_pose(const pose4& value) noexcept;
vec3f world_to_local(const pose4& anchor, vec3f world) noexcept;
pose local_pose(const pose4& anchor, const pose& world_pose) noexcept;
 pose4 relative_pose(const pose4& left, const pose4& right) noexcept;

}
