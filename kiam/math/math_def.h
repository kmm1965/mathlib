#pragma once

#ifndef MATH_USE_CPP17
#if __cplusplus >= 201703 || _HAS_CXX17
#define MATH_USE_CPP17 1
#endif
#endif

#define _KIAM_MATH_BEGIN namespace kiam { namespace math {
#define _KIAM_MATH_END } }

#define _KIAM_MATH ::kiam::math

#ifdef __CUDACC__

#define __DEVICE __device__
#define __HOST __host__
#define __CONSTANT __constant__

#else	// __CUDACC__

#define __DEVICE
#define __HOST
#define __CONSTANT

#endif	// __CUDACC__

#define __DEVICE__HOST
#define __DEVICE_
#define CONSTEXPR constexpr

#ifndef MAX_ARRAY_DIM_SIZE
#define MAX_ARRAY_DIM_SIZE	10
#endif

#ifndef MAX_ASSIGNMENT_SIZE
#define MAX_ASSIGNMENT_SIZE	10
#endif

#define BOOST_PP_ENUM_print_data(z, n, data) data
