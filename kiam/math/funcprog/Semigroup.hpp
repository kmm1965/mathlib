#pragma once

#include "define_function.h"

_FUNCPROG_BEGIN

// requires operator%
template<typename S>
struct Semigroup;

template<typename T>
using Semigroup_t = Semigroup<base_class_t<T> >;

template<class S>
struct is_semigroup : std::false_type {};

template<class S>
using is_semigroup_t = is_semigroup<base_class_t<S> >;

template<class S1, class S2>
struct is_same_semigroup : std::false_type {};

template<class S1, class S2>
using is_same_semigroup_t = is_same_semigroup<base_class_t<S1>, base_class_t<S2> >;

#define DECLARE_SEMIGROUP_CLASS(S) \
    template<typename A> \
    static S<A> semigroup_op(S<A> const& x, S<A> const& y); \
    template<typename A> \
    static S<A> stimes(int n, S<A> const&);

#define IMPLEMENT_SEMIGROUP(S, _S) \
    template<> struct is_semigroup<_S> : std::true_type {}; \
    template<> struct is_same_semigroup<_S, _S> : std::true_type {};

template<typename S>
using semigroup_type = typename std::enable_if<is_semigroup_t<S>::value, S>::type;

template<typename S>
semigroup_type<S> operator%(S const& x, S const& y) {
    return Semigroup_t<S>::semigroup_op(x, y);
}

DEFINE_FUNCTION_2(1, semigroup_type<T0>, stimes, int, n, T0 const&, s,
    return Semigroup_t<T0>::stimes(n, s);)

_FUNCPROG_END
