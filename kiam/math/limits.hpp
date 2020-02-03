#pragma once

#include "math_def.h"

#ifndef __CUDACC__

#include <limits>

#endif  // __CUDACC__

_KIAM_MATH_BEGIN

#ifdef __CUDACC__

template<typename T>
struct numeric_traits_base;

template<>
struct numeric_traits_base<double>
{
    typedef unsigned long long value_type;
    static const unsigned mantissa_length = 52;
};

template<>
struct numeric_traits_base<float>
{
    typedef unsigned value_type;
    static const unsigned mantissa_length = 23;
};

template<typename T>
struct numeric_traits : numeric_traits_base<T>
{
    typedef numeric_traits type;
    typedef numeric_traits_base<T> super;
    typedef typename super::value_type value_type;

    static const size_t length = sizeof(T) * 8;
    static const size_t exp_length = length - super::mantissa_length - 1;
    static const int max_exponent = 1 << (exp_length - 1);
    static const int min_exponent = -max_exponent + 3;

    static const value_type sign_mask       = (value_type) 1 << (length - 1);
    static const value_type mantissa_mask   = ((value_type) 1 << super::mantissa_length) - 1;
    static const value_type exp_mask0       = ((value_type) 1 << exp_length) - 1;
    static const value_type exp_mask        = exp_mask0 << super::mantissa_length;
    static const value_type max_exp_mask    = exp_mask0 << super::mantissa_length;
    static const value_type max_exp_value   = (exp_mask0 - 1) << super::mantissa_length;
    static const value_type exp_value_0     = ((value_type) max_exponent - 2) << super::mantissa_length;

    static const value_type min             = ((value_type) 1) << super::mantissa_length;
    static const value_type max             = max_exp_value | mantissa_mask;
    static const value_type epsilon         = ((value_type) max_exponent - 1 - super::mantissa_length) << super::mantissa_length;
    static const value_type infinity        = exp_mask;
    static const value_type minus_infinity  = infinity | sign_mask;
    static const value_type quiet_NaN       = exp_mask | ((value_type) 1 << (super::mantissa_length - 1));
    static const value_type signaling_NaN   = quiet_NaN + 1;
};

template<typename T>
struct numeric_limits
{
    __device__ __host__
    static T min()
    {
        typedef numeric_traits<T> traits_type;
        typename traits_type::value_type value = traits_type::min;
        return *(T*) &value;
    }

    __device__ __host__
    static T max()
    {
        typedef numeric_traits<T> traits_type;
        typename traits_type::value_type value = traits_type::max;
        return *(T*) &value;
    }

    __device__ __host__
    static T epsilon()
    {
        typedef numeric_traits<T> traits_type;
        typename traits_type::value_type value = traits_type::epsilon;
        return *(T*) &value;
    }

    __device__ __host__
    static T infinity()
    {
        typedef numeric_traits<T> traits_type;
        typename traits_type::value_type value = traits_type::infinity;
        return *(T*) &value;
    }

    __device__ __host__
    static T quiet_NaN()
    {
        typedef numeric_traits<T> traits_type;
        typename traits_type::value_type value = traits_type::quiet_NaN;
        return *(T*) &value;
    }

    __device__ __host__
    static T signaling_NaN()
    {
        typedef numeric_traits<T> traits_type;
        typename traits_type::value_type value = traits_type::signaling_NaN;
        return *(T*) &value;
    }
};

#else   // __CUDACC__

template<typename T>
#ifndef DONT_USE_CXX_11
using numeric_limits = std::numeric_limits<T>;
#else
struct numeric_limits : std::numeric_limits<T>{};
#endif

#endif  // __CUDACC__

_KIAM_MATH_END
