#pragma once

#include "../funcprog_common.hpp"

_FUNCPROG_BEGIN

template<class A>
struct _is_alternative : std::false_type {};

template<class A>
constexpr bool _is_alternative_v = _is_alternative<A>::value;

template<class A>
using is_alternative = _is_alternative<base_class_t<A> >;

template<class A>
constexpr bool is_alternative_v = is_alternative<A>::value;

template<class A, typename T = A>
using alternative_type = std::enable_if_t<is_alternative<A>::value, T>;

template<class _A1, class _A2>
struct _is_same_alternative : std::false_type {};

template<class _A>
struct _is_same_alternative<_A, _A> : _is_alternative<_A> {};

template<class A1, class A2>
using is_same_alternative = _is_same_alternative<base_class_t<A1>, base_class_t<A2> >;

template<class A1, class A2, typename T>
using same_alternative_type = std::enable_if_t<is_same_alternative<A1, A2>::value, T>;

// Requires _empty, operator|
template<typename ALT>
struct Alternative;

template<typename A>
using Alternative_t = Alternative<base_class_t<A> >;

#define IMPLEMENT_ALTERNATIVE(_ALT) \
    template<> struct _is_alternative<_ALT> : std::true_type {}

template<class ALT>
constexpr alternative_type<ALT> operator|(ALT const& op1, ALT const& op2);

_FUNCPROG_END
