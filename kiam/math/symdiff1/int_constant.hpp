#pragma once

#include "expression.hpp"

_SYMDIFF1_BEGIN

template<int N>
struct int_constant : expression<int_constant<N> >
{
    static const int value = N;

    typedef int_constant<0> diff_type;

    constexpr diff_type diff() const
    {
        return diff_type();
    }

    template<typename T>
    constexpr int operator()(T const& x) const {
        return value;
    }

    constexpr std::string to_string() const
    {
        std::ostringstream ss;
        ss << value;
        return ss.str();
    }
};

template<class E>
struct is_int_constant : std::false_type{};

template<int N>
struct is_int_constant<int_constant<N> > :
    std::true_type{};

template<class E>
constexpr bool is_int_constant_v = is_int_constant<E>::value;

template<class E>
struct int_constant_value :
    std::integral_constant<int, 0>{};

template<int N>
struct int_constant_value<int_constant<N> > :
    std::integral_constant<int, N>{};

template<class E>
constexpr int int_constant_val = int_constant_value<E>::value;

_SYMDIFF1_END
