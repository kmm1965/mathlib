#pragma once

#include <type_traits>

#include "generic_operators.hpp"

_KIAM_MATH_BEGIN

template<typename T>
struct get_scalar_type
{
    typedef T type;
};

template<typename T>
using get_scalar_type_t = typename get_scalar_type<T>::type;

template<typename T1, typename T2>
struct supports_multiplies : std::false_type {};

template<typename T1, typename T2>
struct multiplies_result_type;

template<typename T1, typename T2>
using multiplies_result_type_t = typename multiplies_result_type<T1, T2>::type;

#ifdef _HAS_CPP_CONCEPTS

template<typename T>
concept Arithmetic = std::is_arithmetic<T>::value;

template<Arithmetic T>
struct supports_multiplies<T, T> : std::true_type {};

template<Arithmetic T>
struct multiplies_result_type<T, T> { typedef T type; };

#else // _HAS_CPP_CONCEPTS

#define DEFAULT_MULTIPLIES_RESULT_TYPE(T) \
    template<> \
    struct supports_multiplies<T, T> : std::true_type {}; \
    template<> \
    struct multiplies_result_type<T, T>{ typedef T type; }

DEFAULT_MULTIPLIES_RESULT_TYPE(float);
DEFAULT_MULTIPLIES_RESULT_TYPE(double);
DEFAULT_MULTIPLIES_RESULT_TYPE(long double);

#endif // _HAS_CPP_CONCEPTS

template<typename T1, typename T2>
using get_generic_multiplies = generic_multiplies<T1, T2, multiplies_result_type_t<T1, T2> >;

template<typename T1, typename T2>
struct supports_divides : std::false_type {};

template<typename T1, typename T2>
struct divides_result_type;

template<typename T1, typename T2>
using divides_result_type_t = typename divides_result_type<T1, T2>::type;

#ifdef _HAS_CPP_CONCEPTS

template<Arithmetic T>
struct supports_divides<T, T> : std::true_type {};

template<Arithmetic T>
struct divides_result_type<T, T> { typedef T type; };

#else // _HAS_CPP_CONCEPTS

#define DEFAULT_DIVIDES_RESULT_TYPE(T) \
    template<> \
    struct supports_divides<T, T> : std::true_type {}; \
    template<> \
    struct divides_result_type<T, T>{ typedef T type; }

DEFAULT_DIVIDES_RESULT_TYPE(float);
DEFAULT_DIVIDES_RESULT_TYPE(double);
DEFAULT_DIVIDES_RESULT_TYPE(long double);

#endif // _HAS_CPP_CONCEPTS

template<typename T1, typename T2>
using get_generic_divides = generic_divides<T1, T2, divides_result_type_t<T1, T2> >;

template<typename T>
struct supports_scalar_product : std::false_type {};

template<typename T>
struct supports_component_product : std::false_type {};

#ifdef __CUDACC__

template<>
struct get_scalar_type<float2>
{
    typedef float type;
};

template<>
struct supports_multiplies<float2, float2> : std::true_type{};

template<>
struct multiplies_result_type<float2, float2>
{
    typedef float2 type;
};

template<>
struct supports_divides<float2, float2> : std::true_type {};

template<>
struct divides_result_type<float2, float2>
{
    typedef float2 type;
};

template<>
struct supports_divides<float, float2> : std::true_type {};

template<>
struct divides_result_type<float, float2>
{
    typedef float2 type;
};

template<>
struct get_scalar_type<double2>
{
    typedef double type;
};

template<>
struct supports_divides<double2, double2> : std::true_type {};

template<>
struct divides_result_type<double2, double2>
{
    typedef double2 type;
};

template<>
struct supports_divides<double, double2> : std::true_type{};

template<>
struct divides_result_type<double, double2>
{
    typedef double2 type;
};

#else   // __CUDACC__

#ifdef MATH_HAS_COMPLEX
template<typename T>
struct get_scalar_type<std::complex<T> >
{
    typedef T type;
};

template<typename T>
struct supports_multiplies<std::complex<T>, std::complex<T> > : std::true_type {};

template<typename T>
struct multiplies_result_type<std::complex<T>, std::complex<T> >
{
    typedef std::complex<T> type;
};

template<typename T>
struct supports_divides<std::complex<T>, std::complex<T> > : std::true_type {};

template<typename T>
struct divides_result_type<std::complex<T>, std::complex<T> >
{
    typedef std::complex<T> type;
};

template<typename T>
struct supports_divides<T, std::complex<T> > : std::true_type {};

template<typename T>
struct divides_result_type<T, std::complex<T> >
{
    typedef std::complex<T> type;
};
#endif // MATH_HAS_COMPLEX

#endif  // __CUDACC__

typedef typename std::make_signed<size_t>::type ssize_t;

template<typename T>
struct is_dimensionless : std::true_type {};

template<typename T>
struct sqrt_result_type
{
    static_assert(is_dimensionless<T>::value, "Should be dimensionless");
    typedef T type;
};

template<typename T>
using sqrt_result_type_t = typename sqrt_result_type<T>::type;

template<typename T>
struct sqr_result_type
{
    static_assert(is_dimensionless<T>::value, "Should be dimensionless");
    typedef T type;
};

template<typename T>
using sqr_result_type_t = typename sqr_result_type<T>::type;

template<typename T>
struct get_value_type {
    typedef typename T::value_type type;
};

template<typename T>
using get_value_type_t = typename get_value_type<T>::type;

_KIAM_MATH_END
