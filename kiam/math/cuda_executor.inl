#pragma once

#include "context.hpp"
#include "kiam_math_alg.h"

_KIAM_MATH_BEGIN

#ifndef BLOCK1_SIZE
#define BLOCK1_SIZE 512
#endif

template<class Closure>
__global__
void cuda_exec(size_t size, Closure closure)
{
    size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size)
        closure(i);
}

template<typename TAG>
template<class Closure>
void cuda_executor<TAG>::operator()(size_t size, Closure const& closure) const
{
    CudaSynchronize sync;
    unsigned const
        dimGrid = unsigned((size + BLOCK1_SIZE - 1) / BLOCK1_SIZE),
        dimBlock = unsigned(size <= BLOCK1_SIZE ? size : BLOCK1_SIZE);
    cuda_exec<<<dimGrid, dimBlock>>>(size, closure);
}

_KIAM_MATH_END
