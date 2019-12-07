#pragma once

#include "context.hpp"
#include "kiam_math_alg.h"

_KIAM_MATH_BEGIN

#ifndef BLOCK1_SIZE
#define BLOCK1_SIZE	512
#endif

template<class Callback>
__global__
void cuda_exec(Callback callback, size_t size)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
		callback[i];
}

template<class Callback>
void cuda_exec_callback(Callback& callback, size_t size)
{
	const unsigned
		dimGrid = unsigned((size + BLOCK1_SIZE - 1) / BLOCK1_SIZE),
		dimBlock = unsigned(size <= BLOCK1_SIZE ? size : BLOCK1_SIZE);
	cuda_exec<<<dimGrid, dimBlock>>>(callback, size);
	CHECK_CUDA_ERROR("cuda_exec");
}

_KIAM_MATH_END
