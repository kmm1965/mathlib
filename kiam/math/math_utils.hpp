#pragma once

#if defined(__CUDACC__)
#include "cuda_math_utils.hpp"
#elif defined(__OPENCL__)
#include "opencl_math_utils.hpp"
#else
#include "serial_math_utils.hpp"
#endif
