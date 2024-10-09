cmake_minimum_required(VERSION 3.30.4)


project(covoxslam VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 23)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(covoxslam_core STATIC src/types.cpp src/json_reader.cpp src/dataset.cpp)
target_include_directories(covoxslam_core PUBLIC include)
if(MSVC)
    target_compile_options(covoxslam_core PUBLIC /EHsc)
endif()

add_executable(covoxslam apps/main.cpp)
 target_link_libraries(covoxslam PRIVATE covoxslam_core)
target_compile_definitions(covoxslam PRIVATE covoxslam_version="${PROJECT_VERSION}")
