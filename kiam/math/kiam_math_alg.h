#pragma once

#include <algorithm>
#include "math_def.h"

#ifdef __CUDACC__

#include <thrust/version.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#if THRUST_VERSION >= 100900
#include <thrust/system/cuda/detail/util.h>
#define CUDA_THROW_ON_ERROR(error, msg) thrust::cuda_cub::throw_on_error(error, msg)
#else
#include <thrust/system/cuda/detail/throw_on_error.h>
#define CUDA_THROW_ON_ERROR(error, msg) thrust::system::cuda::detail::throw_on_error(error, msg)
#endif
#define CHECK_CUDA_ERROR(msg) CUDA_THROW_ON_ERROR(cudaGetLastError(), msg)

#include <math_constants.h>

#define MATH_STATIC_ASSERT(e) THRUST_STATIC_ASSERT(e)

#define MATH_COPY(first, last, dest) thrust::copy(first, last, dest)
#define MATH_FILL(first, last, val) thrust::fill(first, last, val)
#define MATH_FILL_N(first, count, val) thrust::fill_n(first, count, val)
#define MATH_TRANSFORM(first, last, result, func) thrust::transform(first, last, result, func)
#define MATH_TRANSFORM2(first1, last1, first2, result, func) thrust::transform(first1, last1, first2, result, func)
#define MATH_ACCUMULATE(first, last, val) thrust::reduce(first, last, val)
#define MATH_ACCUMULATE2(first, last, val, bin_op) thrust::reduce(first, last, val, bin_op)
#define MATH_MIN_ELEMENT(first, last) thrust::min_element(first, last)
#define MATH_REDUCE(first, last, val, bin_op) thrust::reduce(first, last, val, bin_op)
#define MATH_INNER_PRODUCT(first1, last1, first2, init) thrust::inner_product(first1, last1, first2, init)
#define MATH_INNER_PRODUCT2(first1, last1, first2, init, bin_op1, bin_op2) thrust::inner_product(first1, last1, first2, init, bin_op1, bin_op2)
#define MATH_FIND(first, last, val) thrust::find(first, last, val)
#define MATH_FIND_IF(first, last, pred) thrust::find_if(first, last, pred)
#define MATH_FOR_EACH(first, last, func) thrust::for_each(first, last, func)
#define MATH_DISTANCE(first, last) thrust::distance(first, last)

_KIAM_MATH_BEGIN

struct CudaSynchronize {
	~CudaSynchronize(){
		CHECK_CUDA_ERROR("synchronize");
		CUDA_THROW_ON_ERROR(cudaDeviceSynchronize(), "synchronize");
	}
};

_KIAM_MATH_END

#elif defined(__OPENCL__)

#include <boost/compute/algorithm/copy.hpp>
#include <boost/compute/algorithm/fill.hpp>
#include <boost/compute/algorithm/fill_n.hpp>
#include <boost/compute/algorithm/transform.hpp>
#include <boost/compute/algorithm/reduce.hpp>
#include <boost/compute/algorithm/inner_product.hpp>
#include <boost/compute/algorithm/find.hpp>
#include <boost/compute/algorithm/find_if.hpp>

#include <boost/static_assert.hpp>
#define MATH_STATIC_ASSERT(e) BOOST_STATIC_ASSERT(e)

#define MATH_COPY(first, last, dest) boost::compute::copy(first, last, dest)
#define MATH_FILL(first, last, val) boost::compute::fill(first, last, val)
#define MATH_FILL_N(first, count, val) boost::compute::fill_n(first, count, val)
#define MATH_TRANSFORM(first, last, result, func) boost::compute::transform(first, last, result, func)
#define MATH_TRANSFORM2(first1, last1, first2, result, func) boost::compute::transform(first1, last1, first2, result, func)
#define MATH_ACCUMULATE(first, last, val) boost::compute::reduce(first, last, val)
#define MATH_ACCUMULATE2(first, last, val, bin_op) boost::compute::reduce(first, last, val, bin_op)
#define MATH_MIN_ELEMENT(first, last) boost::compute::min_element(first, last)
#define MATH_REDUCE(first, last, val, bin_op) boost::compute::reduce(first, last, val, bin_op)
#define MATH_INNER_PRODUCT(first1, last1, first2, init) boost::compute::inner_product(first1, last1, first2, init)
#define MATH_INNER_PRODUCT2(first1, last1, first2, init, bin_op1, bin_op2) boost::compute::inner_product(first1, last1, first2, init, bin_op1, bin_op2)
#define MATH_FIND(first, last, val) boost::compute::find(first, last, val)
#define MATH_FIND_IF(first, last, pred) boost::compute::find_if(first, last, pred)
#define MATH_FOR_EACH(first, last, func) boost::compute::for_each(first, last, func)
#define MATH_DISTANCE(first, last) boost::compute::distance(first, last)

_KIAM_MATH_BEGIN

struct OpenCLSynchronize {
    ~OpenCLSynchronize(){
        boost::compute::system::default_queue().finish();
    }
};

_KIAM_MATH_END

#else // __CUDACC__

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>

#include <functional>
#include <numeric>

#include <vector>

#include <boost/static_assert.hpp>
#define MATH_STATIC_ASSERT(e) BOOST_STATIC_ASSERT(e)

#define MATH_COPY(first, last, dest) std::copy(first, last, dest)
#define MATH_FILL(first, last, val) std::fill(first, last, val)
#define MATH_FILL_N(first, count, val) std::fill_n(first, count, val)
#define MATH_TRANSFORM(first, last, result, func) std::transform(first, last, result, func)
#define MATH_TRANSFORM2(first1, last1, first2, result, func) std::transform(first1, last1, first2, result, func)
#define MATH_ACCUMULATE(first, last, init) std::accumulate(first, last, init)
#define MATH_ACCUMULATE2(first, last, init, bin_op) std::accumulate(first, last, init, bin_op)
#define MATH_MIN_ELEMENT(first, last) std::min_element(first, last)
#define MATH_REDUCE(first, last, init, bin_op) std::accumulate(first, last, init, bin_op)
#define MATH_INNER_PRODUCT(first1, last1, first2, init) std::inner_product(first1, last1, first2, init)
#define MATH_INNER_PRODUCT2(first1, last1, first2, init, bin_op1, bin_op2) std::inner_product(first1, last1, first2, init, bin_op1, bin_op2)
#define MATH_FIND(first, last, val) std::find(first, last, val)
#define MATH_FIND_IF(first, last, pred) std::find_if(first, last, pred)
#define MATH_FOR_EACH(first, last, func) std::for_each(first, last, func)
#define MATH_DISTANCE(first, last) std::distance(first, last)

#endif // __CUDACC__

#define MATH_COPY_N(first, count, dest) MATH_COPY(first, (first) + (count), dest)
