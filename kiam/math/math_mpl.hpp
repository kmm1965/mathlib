#pragma once

#if defined(__CUDACC__)
#include <thrust/functional.h>
#elif defined(__OPENCL__)
#else
#include <functional>
#endif

#include "math_def.h"

_KIAM_MATH_BEGIN

#if defined(__CUDACC__)

template<typename T>
using plus = thrust::plus<T>;

template<typename T>
using minus = thrust::minus<T>;

template<typename T>
using multiplies = thrust::multiplies<T>;

template<typename T>
using divides = thrust::divides<T>;

template<typename T>
using negate = thrust::negate<T>;

template<typename T>
using maximum = thrust::maximum<T>;

template<typename T>
using minimum = thrust::minimum<T>;

template<typename T>
using equal_to = thrust::equal_to<T>;

template<typename T>
using not_equal_to = thrust::not_equal_to<T>;

template<typename T>
using less = thrust::less<T>;

template<typename T>
using less_equal = thrust::less_equal<T>;

template<typename T>
using greater = thrust::greater<T>;

template<typename T>
using greater_equal = thrust::greater_equal<T>;

template<typename T>
using logical_and = thrust::logical_and<T>;

template<typename T>
using logical_or = thrust::logical_or<T>;

template<typename T>
using logical_not = thrust::logical_not<T>;

#elif defined(__OPENCL__)

template<class _Super>
struct opencl_binary_function : _Super
{
    typedef typename _Super::result_type result_type;
    typedef result_type first_argument_type;
    typedef result_type second_argument_type;
};

template<typename T>
using plus = opencl_binary_function<boost::compute::minus<T> >;

template<typename T>
using minus = opencl_binary_function<boost::compute::minus<T> >;

template<typename T>
using multiplies = opencl_binary_function<boost::compute::multiplies<T> >;

template<typename T>
using divides = opencl_binary_function<::boost::compute::divides<T> >;

//template<typename T>
//using negate = boost::compute::unary_minus<T>;

template<typename T>
using equal_to = boost::compute::equal_to<T>;

template<typename T>
using not_equal_to : boost::compute::not_equal_to<T>;

template<typename T>
using logical_and = boost::compute::logical_and<T>;

template<typename T>
using logical_or = boost::compute::logical_or<T>;

template<typename T>
using logical_not = boost::compute::logical_not<T>;

#else	// __CUDACC__

template<typename T>
using plus = std::plus<T>;

template<typename T>
using minus = std::minus<T>;

template<typename T>
using multiplies = std::multiplies<T>;

template<typename T>
using divides = std::divides<T>;

template<typename T>
using negate = std::negate<T>;

template<typename T>
struct maximum
{
    using first_argument_type = T;
    using second_argument_type = T;
    using result_type = T;
    
    const T& operator()(const T& x, const T& y) const {
		return x < y ? y : x;
	}
};

template<typename T>
struct minimum
{
    using first_argument_type = T;
    using second_argument_type = T;
    using result_type = T;

    const T& operator()(const T& x, const T& y) const {
		return x < y ? x : y;
	}
};

template<typename T>
using equal_to = std::equal_to<T>;

template<typename T>
using not_equal_to = std::not_equal_to<T>;

template<typename T>
using less = std::less<T>;

template<typename T>
using less_equal = std::less_equal<T>;

template<typename T>
using greater = std::greater<T>;

template<typename T>
using greater_equal = std::greater_equal<T>;

template<typename T>
using logical_and = std::logical_and<T>;

template<typename T>
using logical_or = std::logical_or<T>;

template<typename T>
using logical_not = std::logical_not<T>;

#endif	// __CUDACC__

template<typename T>
struct maxabs
{
    using first_argument_type = T;
    using second_argument_type = T;
    using result_type = T;

    __DEVICE __HOST
    T operator()(const T &_Left, const T &_Right) const
	{
		const T l = (_Left < 0 ? -_Left : _Left), r = (_Right < 0 ? -_Right : _Right);
		return l < r ? r : l;
	}
};

template<typename T>
struct sumsqr
{
    using first_argument_type = T;
    using second_argument_type = T;
    using result_type = T;

    __DEVICE __HOST
    CONSTEXPR T operator()(const T &_Left, const T &_Right) const {
		return _Left + _Right * _Right;
	}
};

_KIAM_MATH_END
