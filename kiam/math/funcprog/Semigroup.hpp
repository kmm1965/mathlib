#pragma once

#include "define_function.h"

_FUNCPROG_BEGIN

template<class _S>
struct _is_semigroup : std::false_type {};

template<class S>
struct is_semigroup : _is_semigroup<base_class_t<S> > {};

template<typename S, typename T = S>
using semigroup_type = typename std::enable_if<is_semigroup<S>::value, T>::type;

#define IMPLEMENT_SEMIGROUP(_S) \
    template<> struct _is_semigroup<_S> : std::true_type {}

#define IMPLEMENT_SEMIGROUP_COND(S) \
    template<typename A> struct is_semigroup<S<A> > : is_semigroup<A> {}

#define DECLARE_SEMIGROUP_CLASS(S) \
    template<typename A> \
    static constexpr S<A> semigroup_op(S<A> const& x, S<A> const& y); \
    template<typename A> \
    static constexpr S<A> stimes(int n, S<A> const&);

// requires operator%
template<typename S>
struct Semigroup;

template<typename T>
using Semigroup_t = Semigroup<base_class_t<T> >;

template<typename S>
constexpr semigroup_type<S> operator%(S const& x, S const& y) {
    return Semigroup_t<S>::semigroup_op(x, y);
}

DEFINE_FUNCTION_2(1, semigroup_type<T0>, stimes, int, n, T0 const&, s,
    return Semigroup_t<T0>::stimes(n, s);)

_FUNCPROG_END
