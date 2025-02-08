#pragma once

#include "../funcprog2_common.hpp"

_FUNCPROG2_BEGIN

template<class _M>
struct _is_monoid : std::false_type {};

template<class _M>
bool constexpr _is_monoid_v = _is_monoid<_M>::value;

template<class M>
struct is_monoid : _is_monoid<base_class_t<M> > {};

template<class M>
bool constexpr is_monoid_v = is_monoid<M>::value;

template<typename M, typename T = M>
using monoid_type = std::enable_if_t<is_monoid_v<M>, T>;

#define MONOID_TYPE_(M, T) BOOST_IDENTITY_TYPE((monoid_type<M, T>))
#define MONOID_TYPE(M, T) typename MONOID_TYPE_(M, T)

template<class _M1, class _M2>
struct _is_same_monoid : std::false_type {};

template<class _M>
struct _is_same_monoid<_M, _M> : _is_monoid<_M> {};

template<class _M1, class _M2>
bool constexpr _is_same_monoid_v = _is_same_monoid<_M1, _M2>::value;

template<class M1, class M2>
using is_same_monoid = _is_same_monoid<base_class_t<M1>, base_class_t<M2> >;

template<class M1, class M2>
bool constexpr is_same_monoid_v = is_same_monoid<M2, M2>::value;

template<class M1, class M2, typename T>
using same_monoid_type = std::enable_if_t<is_same_monoid_v<M1, M2>, T>;

#define SAME_MONOID_TYPE_(M1, M2, T) BOOST_IDENTITY_TYPE((same_monoid_type<M1, M2, T>))
#define SAME_MONOID_TYPE(M1, M2, T) typename SAME_MONOID_TYPE_(M1, M2, T)

// requires mempty, mappend
template<typename _M>
struct Monoid;

template<typename T>
using Monoid_t = Monoid<base_class_t<T> >;

#define IMPLEMENT_MONOID(_M) \
    template<> struct _is_monoid<_M> : std::true_type {}

DECLARE_FUNCTION_2(1, monoid_type<T0>, mappend, T0 const&, T0 const&);

/*
-- | Fold a list using the monoid.
--
-- For most types, the default definition for 'mconcat' will be
-- used, but the function is included in the class definition so
-- that an optimized version can be provided for specific types.
mconcat :: [a] -> a
mconcat = foldr mappend mempty
*/

template<typename A> struct List;

template<typename T>
constexpr monoid_type<T> mconcat(List<T> const& m);

_FUNCPROG2_END
