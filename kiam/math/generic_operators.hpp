#pragma once

#include "math_mpl.hpp"

_KIAM_MATH_BEGIN

template<typename A1, typename A2, typename R>
struct generic_multiplies
{
    using first_argument_type = A1;
    using second_argument_type = A2;
    using result_type = R;

    __DEVICE __HOST
    CONSTEXPR result_type operator()(const first_argument_type& left, const second_argument_type& right) const {
        return left * right;
    };
};

template<typename A1, typename A2, typename R>
struct generic_divides
{
    using first_argument_type = A1;
    using second_argument_type = A2;
    using result_type = R;

    __DEVICE __HOST
    CONSTEXPR result_type operator()(const first_argument_type& left, const second_argument_type& right) const {
        return left / right;
    };
};

template<typename T>
struct generic_scalar_product
{
    using first_argument_type = T;
    using second_argument_type = T;
    using result_type = typename T::value_type;

    __DEVICE __HOST
    CONSTEXPR result_type operator()(const first_argument_type &left, const second_argument_type &right) const {
        return left & right;
    };
};

template<typename T>
struct generic_component_product
{
    using first_argument_type = T;
    using second_argument_type = T;
    using result_type = T;
    
    __DEVICE __HOST
    CONSTEXPR result_type operator()(const first_argument_type &left, const second_argument_type &right) const {
        return left ^ right;
    };
};

struct math_plus_value
{
    template<typename T>
    struct apply {
        typedef plus<T> type;
    };
};

struct math_minus_value
{
    template<typename T>
    struct apply {
        typedef minus<T> type;
    };
};

struct math_multiplies_value
{
    template<typename T>
    struct apply {
        typedef multiplies<T> type;
    };
};

struct math_divides_value
{
    template<typename T>
    struct apply {
        typedef divides<T> type;
    };
};

_KIAM_MATH_END
