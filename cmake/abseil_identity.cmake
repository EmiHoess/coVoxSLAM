file(GLOB_RECURSE covoxslam_solver_files CONFIGURE_DEPENDS
    RELATIVE "${covoxslam_solver_source}"
    "${covoxslam_solver_source}/include/abseil/*.hpp"
    "${covoxslam_solver_source}/src/*.hpp"
    "${covoxslam_solver_source}/src/*.cpp"
    "${covoxslam_solver_source}/src/*.h"
    "${covoxslam_solver_source}/src/*.cuh"
    "${covoxslam_solver_source}/src/*.cu")
list(APPEND covoxslam_solver_files CMakeLists.txt cmake/abseilConfig.cmake.in)
list(SORT covoxslam_solver_files)
set(covoxslam_solver_identity "")
foreach(source IN LISTS covoxslam_solver_files)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${covoxslam_solver_source}/${source}")
    file(SHA256 "${covoxslam_solver_source}/${source}" hash)
    string(APPEND covoxslam_solver_identity "${source}:${hash}\n")
endforeach()
string(SHA256 covoxslam_solver_hash "${covoxslam_solver_identity}")
file(GENERATE OUTPUT "${PROJECT_BINARY_DIR}/covoxslam_pgo_identity.txt"
    CONTENT "solver=abseil\nsource_sha256=${covoxslam_solver_hash}\ncuda=${covoxslam_enable_cuda}\nceres=OFF\n")
