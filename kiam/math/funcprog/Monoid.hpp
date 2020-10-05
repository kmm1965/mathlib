#pragma once

#include "Semigroup.hpp"
#include "List_fwd.hpp"

_FUNCPROG_BEGIN

// requires mempty, mappend
template<typename M>
struct Monoid;

template<typename T>
using Monoid_t = Monoid<base_class_t<T> >;

template<class M>
struct is_monoid : std::false_type {};

template<class M1, class M2>
struct is_same_monoid : std::false_type {};

template<typename M>
using monoid_type = typename std::enable_if<is_monoid<M>::value, M>::type;

DEFINE_FUNCTION_2(1, monoid_type<T0>, mappend, T0 const&, x, T0 const&, y,
    return Monoid_t<T0>::mappend(x, y);)

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
monoid_type<T> mconcat(List<T> const& m) {
    return Monoid_t<T>::mconcat(m);
}

struct _Monoid
{
    // Default implementation of mappend
    // mappend = (<>)
    template<typename M>
    static typename std::enable_if<is_monoid<M>::value, M>::type mappend(M const& x, M const& y){
        return x % y;
    }

    // Default implementation of mconcat
    // mconcat :: [a] -> a
    // mconcat = foldr mappend mempty
    template<typename M>
    static typename std::enable_if<is_monoid<M>::value, M>::type mconcat(List<M> const& m){
        return foldr(_(mappend<M>), Monoid_t<M>::template mempty<value_type_t<M> >(), m);
    }
};

_FUNCPROG_END
