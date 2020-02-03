#pragma once

#include <cmath>
#include <type_traits>

#include "math_utils.hpp" // math_pair

#ifdef __CUDACC__
#include "limits.hpp"
#endif

#ifdef __CUDACC__

#include <thrust/pair.h>

#define _REF

#else   // __CUDACC__

#include <utility>  // pair

#define _REF &

#endif  // __CUDACC__

#include "math_def.h"

_KIAM_MATH_BEGIN

namespace func {

template<typename T>
__DEVICE __HOST
CONSTEXPR T abs(T x){
    return x < T() ? -x : x;
}

#ifndef min
template<typename T>
__DEVICE __HOST
CONSTEXPR T min(T x, T y){
    return x < y ? x : y;
}
#endif

#ifndef max
template<typename T>
__DEVICE __HOST
CONSTEXPR T max(T x, T y){
    return y < x ? x : y;
}
#endif

template<typename T>
__DEVICE __HOST
CONSTEXPR T sqr(const T &value){
    return value * value;
}

#ifdef __CUDACC__

#define DECLARE_MATH_FUNC(func) \
    __device__ __host__ inline double func(double x){ return ::func(x); } \
    __device__ __host__ inline float func(float x){ return ::func##f(x); } 
#define DECLARE_MATH_FUNC2(func) \
    __device__ __host__ inline double func(double x, double y){ return ::func(x, y); } \
    __device__ __host__ inline float func(float x, float y){ return ::func##f(x, y); } 

#else   // __CUDACC__

#ifdef WIN32
#define DECLARE_MATH_FUNC(func) \
    inline double func(double x){ return std::func(x); } \
    inline float func(float x){ return std::func##f(x); } \
    inline long double func(long double x){ return std::func##l(x); }
#define DECLARE_MATH_FUNC2(func) \
    inline double func(double x, double y){ return std::func(x, y); } \
    inline float func(float x, float y){ return std::func##f(x, y); } \
    inline long double func(long double x, long double y){ return std::func##l(x, y); }
#else
#define DECLARE_MATH_FUNC(func) \
    inline double func(double x){ return std::func(x); } \
    inline float func(float x){ return std::func(x); } \
    inline long double func(long double x){ return std::func(x); }
#define DECLARE_MATH_FUNC2(func) \
    inline double func(double x, double y){ return std::func(x, y); } \
    inline float func(float x, float y){ return std::func(x, y); } \
    inline long double func(long double x, long double y){ return std::func(x, y); }
#endif

#endif  // __CUDACC__

DECLARE_MATH_FUNC(sin)
DECLARE_MATH_FUNC(cos)
DECLARE_MATH_FUNC(tan)
DECLARE_MATH_FUNC(asin)
DECLARE_MATH_FUNC(acos)
DECLARE_MATH_FUNC(atan)
DECLARE_MATH_FUNC(sinh)
DECLARE_MATH_FUNC(cosh)
DECLARE_MATH_FUNC(tanh)
DECLARE_MATH_FUNC(ceil)
DECLARE_MATH_FUNC(floor)
DECLARE_MATH_FUNC(exp)
DECLARE_MATH_FUNC(log)
DECLARE_MATH_FUNC(log10)
DECLARE_MATH_FUNC(sqrt)
DECLARE_MATH_FUNC2(pow)

#ifdef __CUDACC__

DECLARE_MATH_FUNC(sinpi)
DECLARE_MATH_FUNC(cospi)
DECLARE_MATH_FUNC(exp2)
DECLARE_MATH_FUNC(log1p)
DECLARE_MATH_FUNC(log2)

#endif  // __CUDACC__

template<typename T>
__DEVICE __HOST
CONSTEXPR T conj(const T &value){
    return value;
}

template<typename T>
__DEVICE __HOST
CONSTEXPR T abs2(const T &value){
    return func::sqr(func::abs(value));
}

#ifdef __CUDACC__

__device__ __host__
float2 inline conj(const float2 &value)
{
    float2 result = { value.x, -value.y };
    return result;
}

__device__ __host__
double2 inline conj(const double2 &value)
{
    double2 result = { value.x, -value.y };
    return result;
}

__device__ __host__
float inline abs2(const float2 &value){
    return func::sqr(value.x) + func::sqr(value.y);
}

__device__ __host__
double inline abs2(const double2 &value){
    return func::sqr(value.x) + func::sqr(value.y);
}

template<typename T>
__device__ __host__
T frexp(T value, int* exp)
{
    typedef numeric_traits<T> traits;
    typedef typename traits::value_type value_type;

    value_type* val = (value_type*) &value;
    if((*val & ~traits::sign_mask) == 0){
        *exp = 0;
        return value;
    }
    if((*val & traits::exp_mask) == traits::exp_mask){
        *exp = -1;
        return value;
    }
    *exp = int((*val & traits::exp_mask) >> traits::mantissa_length) + traits::min_exponent - 1;
    *val = (*val & (traits::sign_mask | traits::mantissa_mask)) | traits::exp_value_0;
    return value;
}

template<typename T>
__device__ __host__
T ldexp(T value, int exp)
{
    typedef numeric_traits<T> traits;
    typedef typename traits::value_type value_type;
    value_type* val = (value_type*) &value;

    if((*val & ~traits::sign_mask) == 0)
        return value;
    exp += (*val & traits::exp_mask) >> traits::mantissa_length;
    if(exp <= 0)
        return T();
    else if(exp > traits::max_exponent - traits::min_exponent + 1){
        typename traits::value_type inf = value > 0 ? traits::infinity : traits::minus_infinity;
        return *(T*)(&inf);
    }
    *val = (*val & (traits::sign_mask | traits::mantissa_mask)) | ((value_type) exp << traits::mantissa_length);
    return value;
}

#else   // __CUDACC__

#ifdef _COMPLEX_

template<typename T>
CONSTEXPR std::complex<T> conj(const std::complex<T> &value){
    return std::conj(value);
}

template<typename T>
CONSTEXPR T abs2(const std::complex<T> &value){
    return sqr(value.real()) + sqr(value.imag());
}

#endif  // _COMPLEX_

template<typename T>
T frexp(T value, int* exp){
    return std::frexp(value, exp);
}

template<typename T>
T ldexp(T mant, int exp){
    return std::ldexp(mant, exp);
}

#endif  // __CUDACC__

}   // namespace func

template<typename T>
struct abs_bin_op
{
    typedef T data_type;

    __DEVICE __HOST
    CONSTEXPR data_type operator()(data_type result, data_type arg2) const {
        return result + func::abs(arg2);
    }
};

template<typename T>
struct abs2_bin_op
{
    typedef T data_type;

    __DEVICE __HOST
    CONSTEXPR data_type operator()(data_type result, data_type arg) const {
        return result + arg * arg;
    }
};

template<typename T>
struct abs2_scale_bin_op
{
    typedef T data_type;
    typedef math_pair<data_type, data_type> pair_type;

    __DEVICE __HOST
    pair_type operator()(const pair_type &pair, data_type arg) const
    {
        if(arg == data_type())
            return pair;
        data_type x = func::abs(arg);
        data_type
            result = pair.first,
            scale = pair.second;
        if(scale < x){
            result = 1 + result * func::sqr(scale / x);
            scale = x;
        } else result += func::sqr(x / scale);
        return pair_type(result, scale);
    }
};

template<typename T>
struct abs2_scale_bin_op2
{
    typedef T data_type;
    typedef math_pair<data_type, data_type> pair_type;

    __DEVICE __HOST
    CONSTEXPR pair_type operator()(const pair_type &pair1, const pair_type &pair2) const
    {
        return pair1.second == pair2.second ?
            pair_type(pair1.first + pair2.first, pair1.second) :
            pair1.second > pair2.second ?
                pair_type(pair1.first + pair2.first * func::sqr(pair2.second / pair1.second), pair1.second) :
                pair_type(pair2.first + pair1.first * func::sqr(pair1.second / pair2.second), pair2.second);
    }
};

struct max_abs_pred
{
    template<typename T>
    __DEVICE __HOST
    CONSTEXPR bool operator()(T x, T y) const {
        return (x >= 0 ? x : -x) < (y >= 0 ? y : -y);
    }
};

struct max_abs_op
{
    template<typename T>
    __DEVICE __HOST
    T operator()(T x, T y) const
    {
        if(x < 0) x = -x;
        if(y < 0) y = -y;
        return x >= y ? x : y;
    }
};

struct ldexp_op
{
    __DEVICE __HOST
    CONSTEXPR ldexp_op(int e_) : e(e_){}

    template<typename T>
    __DEVICE __HOST
    CONSTEXPR T operator()(T x) const {
        return ldexp(x, e);
    }

private:
    int e;
};

_KIAM_MATH_END

#ifdef BOOST_UNITS_QUANTITY_HPP

#include <boost/units/cmath.hpp>

_KIAM_MATH_BEGIN

namespace func {

template<class Unit, class Y>
__DEVICE __HOST
CONSTEXPR typename boost::units::root_typeof_helper<boost::units::quantity<Unit, Y>, boost::units::static_rational<2> >::type sqrt(const boost::units::quantity<Unit, Y>& val) {
    return boost::units::sqrt(val);
}

template<class Unit, class Y>
__DEVICE __HOST
CONSTEXPR boost::units::quantity<typename boost::units::multiply_typeof_helper<Unit, Unit>::type, Y> sqr(const boost::units::quantity<Unit, Y>& val) {
    return val * val;
}

} // namespace func

_KIAM_MATH_END

#endif // BOOST_UNITS_QUANTITY_HPP
