cmake_minimum_required(VERSION 3.30.4)


project(covoxslam VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 23)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(covoxslam_core STATIC src/types.cpp src/json_reader.cpp src/dataset.cpp)
target_sources(covoxslam_core PRIVATE src/pipeline.cpp)
target_sources(covoxslam_core PRIVATE src/mapping.cpp)
target_sources(covoxslam_core PRIVATE src/export.cpp)
option(covoxslam_enable_cuda "Build CUDA runtime and mapping support" OFF)
if(covoxslam_enable_cuda)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit 12.6 REQUIRED)
    target_sources(covoxslam_core PRIVATE src/cuda_mapping.cpp cuda/foundation.cu)
    target_sources(covoxslam_core PRIVATE cuda/ray_contributions.cu)
    target_sources(covoxslam_core PRIVATE cuda/serial_fusion.cu)
    target_sources(covoxslam_core PRIVATE cuda/esdf_site.cu)
    target_link_libraries(covoxslam_core PUBLIC CUDA::cudart)
    set_target_properties(covoxslam_core PROPERTIES CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF CUDA_RUNTIME_LIBRARY Shared)
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/covoxslam_build_metadata.txt"
        CONTENT "cmake=${CMAKE_VERSION}\nhost=${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\ncuda=${CMAKE_CUDA_COMPILER_VERSION}\narchitecture=${CMAKE_CUDA_ARCHITECTURES}\nhost_standard=23\ncuda_standard=20\n")
else()
    target_sources(covoxslam_core PRIVATE src/cuda_mapping_stub.cpp)
endif()
target_include_directories(covoxslam_core PRIVATE cuda)
target_include_directories(covoxslam_core PUBLIC include)
if(MSVC)
    target_compile_options(covoxslam_core PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>")
endif()

add_executable(covoxslam apps/main.cpp)
 target_link_libraries(covoxslam PRIVATE covoxslam_core)
target_compile_definitions(covoxslam PRIVATE covoxslam_version="${PROJECT_VERSION}")
add_executable(covoxslam_cuda apps/cuda_main.cpp)
target_link_libraries(covoxslam_cuda PRIVATE covoxslam_core)
