cmake_minimum_required(VERSION 3.30.4)


project(covoxslam VERSION 0.1.0 LANGUAGES CXX)


set(CMAKE_CXX_STANDARD 23)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(covoxslam apps/main.cpp)
target_compile_definitions(covoxslam PRIVATE covoxslam_version="${PROJECT_VERSION}")
