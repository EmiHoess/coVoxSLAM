#include <covoxslam/cuda.hpp>
#include "sparse_mapping.h"
#include "cuda_site_conversion.hpp"
#include "cuda_blocks_conversion.hpp"

#include <cuda_runtime_api.h>
#include "cuda_blocks_storage.hpp"
#include <array>
#include <algorithm>
#include <vector>
#include <limits>
#include <stdexcept>

#include <type_traits>


namespace covoxslam
{
namespace
{
operation_status cuda_status(cudaError_t error)
{
    if (error != cudaSuccess) return std::unexpected(operation_error{cudaGetErrorString(error)});
    return {};
}

void check_cuda(cudaError_t error)
{
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}


}


#include <algorithm>
#include <chrono>
#include <cmath>
