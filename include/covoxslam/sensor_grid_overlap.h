#pragma once

#include "sensor_grid_sampling.h"

#if defined(__CUDACC__)
#define covoxslam_overlap_callable __host__ __device__
#else
#define covoxslam_overlap_callable
#endif

namespace covoxslam::sensor_grid {

struct overlap_box {
    float center[3]{};
    float axes[3][3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float voxel_size{};
};

namespace overlap_detail {

struct vector {
    double x, y, z;
    covoxslam_overlap_callable double component(int axis) const {
        return axis == 0 ? x : (axis == 1 ? y : z);
    }
};
covoxslam_overlap_callable inline vector add(vector a, vector b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
covoxslam_overlap_callable inline vector sub(vector a, vector b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
covoxslam_overlap_callable inline vector mul(vector a, double s) { return {a.x*s, a.y*s, a.z*s}; }
covoxslam_overlap_callable inline double dot(vector a, vector b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
covoxslam_overlap_callable inline vector cross(vector a, vector b) {
    return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
covoxslam_overlap_callable inline double absolute(double x) { return x < 0 ? -x : x; }

struct geometry {
    vector axis[3];
    vector inverse[3];
    double extent[3];
    bool aligned;
};

covoxslam_overlap_callable inline bool prepare(const overlap_box& box, geometry& geometry) {
    using sampling_detail::finite;
    if (!(box.voxel_size > 0) || !finite(box.voxel_size)) return false;
    for (int a = 0; a < 3; ++a) {
        if (!finite(box.center[a])) return false;
        for (int b = 0; b < 3; ++b) if (!finite(box.axes[a][b])) return false;
        geometry.axis[a] = {box.axes[a][0], box.axes[a][1], box.axes[a][2]};
    }
    constexpr double tolerance = 32.0 * FLT_EPSILON;
    for (int a = 0; a < 3; ++a) {
        if (absolute(dot(geometry.axis[a], geometry.axis[a]) - 1.0) > tolerance) return false;
        for (int b = 0; b < a; ++b)
            if (absolute(dot(geometry.axis[a], geometry.axis[b])) > tolerance) return false;
    }
    const double determinant = dot(geometry.axis[0], cross(geometry.axis[1], geometry.axis[2]));
    if (absolute(determinant - 1.0) > tolerance) return false;
    geometry.aligned = true;
    for (int a = 0; a < 3; ++a) {

        geometry.inverse[a] = mul(cross(geometry.axis[(a+1)%3], geometry.axis[(a+2)%3]), 1.0/determinant);
        geometry.extent[a] = 0;
        for (int b = 0; b < 3; ++b) {
            const double value = absolute(geometry.axis[b].component(a));
            geometry.extent[a] += 0.5 * value;
            if (value != 0.0 && value != 1.0) geometry.aligned = false;
        }
    }
    return true;
}

covoxslam_overlap_callable inline int clip(const vector* input, int count, vector* output,
                                     vector normal, double limit) {
    if (count == 0) return 0;
    int written = 0;
    vector previous = input[count-1];
    double previous_distance = dot(normal, previous) - limit;
    for (int i = 0; i < count; ++i) {
        const vector current = input[i];
        const double distance = dot(normal, current) - limit;
        const bool inside = distance <= 0.0, previous_inside = previous_distance <= 0.0;

        if (inside != previous_inside && distance != 0.0 && previous_distance != 0.0) {
            if (written == 12) return -1;
            output[written++] = add(previous, mul(sub(current, previous),
                previous_distance / (previous_distance - distance)));
        }
        if (inside) {
            if (written == 12) return -1;
            output[written++] = current;
        }
        previous = current;
        previous_distance = distance;
    }
    return written;
}


}
}
#undef covoxslam_overlap_callable
