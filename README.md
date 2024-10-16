## Table of Contents

coVoxSLAM

GPU-accelerated volumetric SLAM system for globally consistent maps for small and large-scale environments.


- [Why coVoxSLAM?]
- [How to use nvblox soon]
  - [C++ Interface soon]
- [Installation] (#installation)
  - [Compilers] (#compilers)
  - [Dependencies] (#deps)
  - [Build] (#build)
  - [Jetson soon]
- [License soon]
- [Paper soon]


## Installation <a id='installation'></a>

coVoxSLAM consists of two parts, the CPU version and the GPU version of the system.

## Compilers <a id='compilers'></a>

To build coVoxSLAM you need a compiler capable of C++20. It has been tested on:

    GCC 11 and above on Linux
    Clang 15 and above on Linux
    MSVC 2019 and above on Windows

## Dependencies <a id='deps'></a>

Dependencies are managed by CMake

  Ceres for the CPU version
  CUDA 12 for the GPU version

## Build

```
mkdir build
cd build
cmake .
```