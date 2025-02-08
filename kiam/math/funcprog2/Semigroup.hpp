#pragma once

#include "fwd/Semigroup_fwd.hpp"
#include "fwd/Monoid_fwd.hpp"

_FUNCPROG2_BEGIN

template<typename _S>
struct _Semigroup // Default implementation of some functions
{
    //-- | This is a valid definition of 'stimes' for an idempotent 'Monoid'.
    template<typename A>
    static constexpr std::enable_if_t<_is_monoid_v<_S>, typeof_t<_S, A> >
    stimesIdempotentMonoid(int n, typeof_t<_S, A> const& x);

    //-- | This is a valid definition of 'stimes' for a 'Monoid'.
    template<typename A>
    static constexpr monoid_type<typeof_t<_S, A> >
    stimes(int n, typeof_t<_S, A> const& x);

    //stimesDefault :: (Integral b, Semigroup a) => b -> a -> a
    template<typename A>
    static constexpr std::enable_if_t<!_is_monoid<_S>::value, typeof_t<_S, A> >
    stimes(int n, typeof_t<_S, A> const& x);
};

#define DECLARE_SEMIGROUP_CLASS(S) \
    template<typename A> \
    static constexpr S<A> sg_op(S<A> const& x, S<A> const& y); \
    template<typename A> \
    static constexpr S<A> stimes(int n, S<A> const&);

_FUNCPROG2_END
