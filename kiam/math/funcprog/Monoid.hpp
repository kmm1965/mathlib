#pragma once

#include "Semigroup.hpp"
#include "List_fwd.hpp"

_FUNCPROG_BEGIN

template<class _M>
struct _is_monoid : std::false_type {};

template<class M>
struct is_monoid : _is_monoid<base_class_t<M> > {};

template<typename M, typename T = M>
using monoid_type = typename std::enable_if<is_monoid<M>::value, T>::type;

template<class _M1, class _M2>
struct _is_same_monoid : std::false_type {};

template<class _M>
struct _is_same_monoid<_M, _M> : _is_monoid<_M> {};

template<class M1, class M2>
using is_same_monoid = _is_same_monoid<base_class_t<M1>, base_class_t<M2> >;

template<class M1, class M2, typename T>
using same_monoid_type = typename std::enable_if<is_same_monoid<M1, M2>::value, T>::type;

#define IMPLEMENT_MONOID(_M) \
    template<> struct _is_monoid<_M> : std::true_type {}

// requires mempty, mappend
template<typename M>
struct Monoid;

template<typename T>
using Monoid_t = Monoid<base_class_t<T> >;

DEFINE_FUNCTION_2(1, constexpr monoid_type<T0>, mappend, T0 const&, x, T0 const&, y,
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
constexpr monoid_type<T> mconcat(List<T> const& m){
    return Monoid_t<T>::mconcat(m);
}

struct _Monoid
{
    // Default implementation of mappend
    // mappend = (<>)
    template<typename M>
    static monoid_type<M> mappend(M const& x, M const& y){
        return x % y;
    }

    // Default implementation of mconcat
    // mconcat :: [a] -> a
    // mconcat = foldr mappend mempty
    template<typename M>
    static constexpr monoid_type<M> mconcat(List<M> const& m){
        return foldr(_(mappend<M>), Monoid_t<M>::template mempty<value_type_t<M> >(), m);
    }
};

_FUNCPROG_END
