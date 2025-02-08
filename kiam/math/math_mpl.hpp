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
using math_plus = thrust::plus<T>;

template<typename T>
using math_minus = thrust::minus<T>;

template<typename T>
using math_multiplies = thrust::multiplies<T>;

template<typename T>
using math_divides = thrust::divides<T>;

template<typename T>
using math_negate = thrust::negate<T>;

template<typename T>
using math_maximum = thrust::maximum<T>;

template<typename T>
using math_minimum = thrust::minimum<T>;

template<typename T>
using math_equal_to = thrust::equal_to<T>;

template<typename T>
using math_not_equal_to = thrust::not_equal_to<T>;

template<typename T>
using math_less = thrust::less<T>;

template<typename T>
using math_less_equal = thrust::less_equal<T>;

template<typename T>
using math_greater = thrust::greater<T>;

template<typename T>
using math_greater_equal = thrust::greater_equal<T>;

using math_logical_and = thrust::logical_and<bool>;
using math_logical_or = thrust::logical_or<bool>;
using math_logical_not = thrust::logical_not<bool>;

#elif defined(__OPENCL__)

template<class _Super>
struct opencl_binary_function : _Super
{
    typedef typename _Super::result_type result_type;
    typedef result_type first_argument_type;
    typedef result_type second_argument_type;
};

template<typename T>
using math_plus = opencl_binary_function<boost::compute::plus<T> >;

template<typename T>
using math_minus = opencl_binary_function<boost::compute::minus<T> >;

template<typename T>
using math_multiplies = opencl_binary_function<boost::compute::multiplies<T> >;

template<typename T>
using math_divides = opencl_binary_function<::boost::compute::divides<T> >;

template<typename T>
using math_negate = boost::compute::unary_negate<T>;

template<typename T>
using math_equal_to = boost::compute::equal_to<T>;

template<typename T>
using math_not_equal_to = boost::compute::not_equal_to<T>;

using math_logical_and = boost::compute::logical_and<void>;
using math_logical_or = boost::compute::logical_or<void>;
using math_logical_not = boost::compute::logical_not<void>;

#else   // __CUDACC__

template<typename T>
using math_plus = std::plus<T>;

template<typename T>
using math_minus = std::minus<T>;

template<typename T>
using math_multiplies = std::multiplies<T>;

template<typename T>
using math_divides = std::divides<T>;

template<typename T>
using math_negate = std::negate<T>;

template<typename T>
struct math_maximum
{
    using first_argument_type = T;
    using second_argument_type = T;
    using result_type = T;
    
    const T& operator()(const T& x, const T& y) const {
        return x < y ? y : x;
    }
};

template<typename T>
struct math_minimum
{
    using first_argument_type = T;
    using second_argument_type = T;
    using result_type = T;

    const T& operator()(const T& x, const T& y) const {
        return x < y ? x : y;
    }
};

template<typename T>
using math_equal_to = std::equal_to<T>;

template<typename T>
using math_not_equal_to = std::not_equal_to<T>;

template<typename T>
using math_less = std::less<T>;

template<typename T>
using math_less_equal = std::less_equal<T>;

template<typename T>
using math_greater = std::greater<T>;

template<typename T>
using math_greater_equal = std::greater_equal<T>;

using math_logical_and = std::logical_and<>;
using math_logical_or = std::logical_or<>;
using math_logical_not = std::logical_not<>;

#endif  // __CUDACC__

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
