#pragma once

#include "../funcprog_common.hpp"

_FUNCPROG_BEGIN

template<class _S>
struct _is_semigroup : std::false_type {};

template<class _S>
bool constexpr _is_semigroup_v = _is_semigroup<_S>::value;

template<class S>
struct is_semigroup : _is_semigroup<base_class_t<S> > {};

template<class S>
bool constexpr is_semigroup_v = is_semigroup<S>::value;

template<typename S, typename T = S>
using semigroup_type = std::enable_if_t<is_semigroup_v<S>, T>;

#define SEMIGROUP_TYPE_(S, T) BOOST_IDENTITY_TYPE((semigroup_type<S, T>))
#define SEMIGROUP_TYPE(S, T) typename SEMIGROUP_TYPE_(S, T)

// requires operator%
template<typename _S>
struct Semigroup;

template<typename T>
using Semigroup_t = Semigroup<base_class_t<T> >;

#define IMPLEMENT_SEMIGROUP(_S) \
    template<> struct _is_semigroup<_S> : std::true_type {}

#define IMPLEMENT_SEMIGROUP_COND(S) \
    template<typename A> struct is_semigroup<S<A> > : is_semigroup<A> {}

DECLARE_FUNCTION_2(1, semigroup_type<T0>, semigroup_op, T0 const&, T0 const&)

template<typename S>
constexpr semigroup_type<S> operator%(S const& x, S const& y){
    return semigroup_op(x, y);
}

DECLARE_FUNCTION_2(1, semigroup_type<T0>, stimes, int, T0 const&);

_FUNCPROG_END
