option(covoxslam_enable_pgo "Link external native pose graph solvers" OFF)
set(covoxslam_ceronoceres_path "../ceronoceres" CACHE STRING "Relative external solver source path")

if(covoxslam_enable_pgo)
    cmake_path(ABSOLUTE_PATH covoxslam_ceronoceres_path BASE_DIRECTORY "${PROJECT_SOURCE_DIR}"
        NORMALIZE OUTPUT_VARIABLE covoxslam_solver_source)
endif()
