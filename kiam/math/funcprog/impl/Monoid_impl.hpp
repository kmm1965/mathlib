#pragma once

#include "../Monoid.hpp"

_FUNCPROG_BEGIN

// Default implementation of mappend
// mappend = (<>)
template<typename _M>
template<typename A>
constexpr monoid_type<A> _Monoid<_M>::mappend(A const& x, A const& y){
    return x % y;
}

// Default implementation of mconcat
// mconcat :: [a] -> a
// mconcat = foldr mappend mempty
template<typename _M>
template<typename A>
constexpr monoid_type<A> _Monoid<_M>::mconcat(List<A> const& m){
    return foldr(_(Monoid_t<A>::template mappend<A>), A::mempty(), m);
}

FUNCTION_TEMPLATE(1) constexpr monoid_type<T0> mappend(T0 const& x, T0 const& y) {
    return Monoid_t<T0>::mappend(x, y);
}

/*
-- | Fold a list using the monoid.
--
-- For most types, the default definition for 'mconcat' will be
-- used, but the function is included in the class definition so
-- that an optimized version can be provided for specific types.
mconcat :: [a] -> a
mconcat = foldr mappend mempty
*/

template<typename T>
constexpr monoid_type<T> mconcat(List<T> const& m){
    return Monoid_t<T>::mconcat(m);
}

_FUNCPROG_END
