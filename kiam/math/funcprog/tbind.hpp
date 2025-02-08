#pragma once

#include "funcprog_setup.h"

_FUNCPROG_BEGIN

template<template<typename...> class OP, typename...Ts>
struct tbind;

namespace detail {

    template<template<typename...> class OP, typename PARAMS, typename...Ts>
    struct tbind_impl;

    template<template<typename...> class OP, typename...Ss>
    struct tbind_impl<OP, std::tuple<Ss...> >
    {
        template<typename...Us>
        using ttype = OP<Ss...>;
    };

    template<template<typename...> class OP, typename...Ss, typename T, typename...Ts>
    struct tbind_impl<OP, std::tuple<Ss...>, T, Ts...>
    {
        template<typename...Us>
        using ttype = typename tbind_impl<OP,
            std::tuple<Ss..., T>, Ts...
        >::template ttype<Us...>;
    };

    template<template<typename...> class OP, typename...Ss, size_t I, typename...Ts>
    struct tbind_impl<OP, std::tuple<Ss...>, std::integral_constant<size_t, I>, Ts...>
    {
        template<typename...Us>
        using ttype = typename tbind_impl<OP,
            typename std::tuple<Ss...,
                typename std::tuple_element<I, typename std::tuple<Us...> >::type
            >, Ts...
        >::template ttype<Us...>;
    };

    template<template<typename...> class OP, typename...Ss, size_t N, size_t... Ns, typename...Ts>
    struct tbind_impl<OP, std::tuple<Ss...>, std::integer_sequence<size_t, N, Ns...>, Ts...>
    {
        template<typename...Us>
        using ttype = typename tbind_impl<OP,
            typename std::tuple<Ss...,
                typename std::tuple_element<N, typename std::tuple<Us...> >::type
            >, std::integer_sequence<size_t, Ns...>, Ts...
        >::template ttype<Us...>;
    };

    template<template<typename...> class OP, typename...Ss, typename...Ts>
    struct tbind_impl<OP, std::tuple<Ss...>, std::integer_sequence<size_t>, Ts...>
    {
        template<typename...Us>
        using ttype = typename tbind_impl<OP,
            typename std::tuple<Ss...>, Ts...
        >::template ttype<Us...>;
    };

} // namespace detail

template<size_t N>
using tn = std::integral_constant<size_t, N>;

template<size_t... Ns>
using tns = std::integer_sequence<size_t, Ns...>;

template<template<typename...> class OP, typename...Ts>
struct tbind : detail::tbind_impl<OP, std::tuple<>, Ts...>{};

template<template<typename...> class OP, typename...Ts>
struct tbind_front : tbind<OP, Ts..., std::integral_constant<size_t, 0> > {};
//struct tbind_front : tbind<OP, Ts..., std::make_index_sequence<sizeof...(Ss) - sizeof...(Ts)> > {};

template<template<typename...> class OP, size_t N>
struct any_to_specific;

template<template<typename...> class OP>
struct any_to_specific<OP, 1>
{
    template<typename T0>
    using ttype = OP<T0>;
};

template<template<typename...> class OP>
struct any_to_specific<OP, 2>
{
    template<typename T0, typename T1>
    using ttype = OP<T0, T1>;
};

template<template<typename...> class OP>
struct any_to_specific<OP, 3>
{
    template<typename T0, typename T1, typename T2>
    using ttype = OP<T0, T1, T2>;
};

template<template<typename...> class OP>
struct any_to_specific<OP, 4>
{
    template<typename T0, typename T1, typename T2, typename T3>
    using ttype = OP<T0, T1, T2, T3>;
};

template<template<typename...> class OP>
struct any_to_specific<OP, 5>
{
    template<typename T0, typename T1, typename T2, typename T3, typename T4>
    using ttype = OP<T0, T1, T2, T3, T4>;
};

_FUNCPROG_END
