#pragma once

#include "Monoid.hpp"

_FUNCPROG_BEGIN

// requires foldl, foldl1, foldr, foldr1
template<typename F>
struct Foldable;

template<typename T>
using Foldable_t = Foldable<base_class_t<T> >;

template<class F>
struct is_foldable : std::false_type {};

#define DECLARE_FOLDABLE_CLASS(F) \
    /* foldMap :: Monoid m => (a -> m) -> t a -> m */ \
    template<typename M, typename Arg> \
    static typename std::enable_if<is_monoid_t<M>::value, M>::type foldMap(function_t<M(Arg)> const& f, F<fdecay<Arg> > const& x); \
    \
    /* foldl :: (b -> a -> b) -> b -> t a -> b */ \
    template<typename Ret, typename A, typename B> \
    static typename std::enable_if<is_same_as<Ret, B>::value, Ret>::type \
    foldl(function_t<Ret(B, A)> const& f, Ret const& z, F<fdecay<A> > const& x); \
    \
    /* foldl1 :: (a -> a -> a) -> t a -> a */ \
    template<typename A, typename Arg1, typename Arg2> \
    static typename std::enable_if<is_same_as<A, Arg1>::value && is_same_as<A, Arg2>::value, A>::type \
    foldl1(function_t<A(Arg1, Arg2)> const& f, F<A> const& x); \
    \
    /* foldr :: (a -> b -> b) -> b -> t a -> b */ \
    template<typename Ret, typename A, typename B> \
    static typename std::enable_if<is_same_as<Ret, B>::value, Ret>::type \
    foldr(function_t<Ret(A, B)> const& f, Ret const& z, F<fdecay<A> > const& x); \
    \
    /* foldr1 :: (a -> a -> a) -> t a -> a */ \
    template<typename A, typename Arg1, typename Arg2> \
    static typename std::enable_if<is_same_as<A, Arg1>::value && is_same_as<A, Arg2>::value, A>::type \
    foldr1(function_t<A(Arg1, Arg2)> const& f, F<A> const& x); \

// foldMap f = foldr (mappend . f) mempty
#define DEFAULT_FOLDMAP_IMPL(F, _F) \
    template<typename M, typename Arg> \
    typename std::enable_if<is_monoid_t<M>::value, M>::type Foldable<_F>::foldMap(function_t<M(Arg)> const& f, F<fdecay<Arg> > const& x){ \
        return foldr(_(mappend<M>) & f, Monoid_t<M>::template mempty<value_type_t<M> >(), x); \
    }

#define IMPLEMENT_FOLDABLE(F) \
    template<typename A> \
    struct is_foldable<F<A> > : std::true_type {};

/*
-- | Map each element of the structure to a monoid,
-- and combine the results.
foldMap :: Monoid m => (a -> m) -> t a -> m
{-# INLINE foldMap #-}
-- This INLINE allows more list functions to fuse. See Trac #9848.
foldMap f = foldr (mappend . f) mempty
*/
template<typename F, typename M, typename Arg>
using foldMap_type = typename std::enable_if<
    is_foldable<F>::value && is_monoid_t<M>::value && is_same_as<value_type_t<F>, Arg>::value,
    M
>::type;

#define FOLDMAP_TYPE_(F, M, Arg) BOOST_IDENTITY_TYPE((foldMap_type<F, M, Arg>))
#define FOLDMAP_TYPE(F, M, Arg) typename FOLDMAP_TYPE_(F, M, Arg)

DEFINE_FUNCTION_2(3, FOLDMAP_TYPE(T0, T1, T2), foldMap, function_t<T1(T2)> const&, f, T0 const&, v,
    return Foldable_t<T0>::foldMap(f, v);)

/*
-- | Combine the elements of a structure using a monoid.
fold :: Monoid m => t m -> m
fold = foldMap id
*/
template<typename F>
using fold_type = typename std::enable_if<
    is_foldable<F>::value && is_monoid_t<value_type_t<F> >::value,
    value_type_t<F>
>::type;

// fold :: Monoid m => t m -> m
// fold = foldMap id
template<typename T>
fold_type<T> fold(T const& v) {
    return foldMap(_(id<value_type_t<T> >), v);
}

template<typename FO, typename A, typename B, typename Ret>
using foldlr_type = typename std::enable_if<
    is_foldable<FO>::value && is_same_as<value_type_t<FO>, A>::value && is_same_as<Ret, B>::value,
    Ret
>::type;

#define FOLDLR_TYPE_(FO, A, B, Ret) BOOST_IDENTITY_TYPE((foldlr_type<FO, A, B, Ret>))
#define FOLDLR_TYPE(FO, A, B, Ret) typename FOLDLR_TYPE_(FO, A, B, Ret)

template<typename FO>
using fold1_type = typename std::enable_if<is_foldable<FO>::value, value_type_t<FO> >::type;

DEFINE_FUNCTION_3(4, FOLDLR_TYPE(T0, T1, T2, T3), foldl, function_t<T3(T2, T1)> const&, f, T3 const&, z, T0 const&, x,
    return Foldable_t<T0>::foldl(f, z, x);)

// foldl1 :: (a -> a -> a) -> t a -> a
DEFINE_FUNCTION_2(4, fold1_type<T0>, foldl1, function_t<T3(T1, T2)> const&, f, T0 const&, x,
    return Foldable_t<T0>::foldl1(f, x);)

// foldr :: (a -> b -> b) -> b -> t a -> b
DEFINE_FUNCTION_3(4, FOLDLR_TYPE(T0, T1, T2, T3), foldr, function_t<T3(T1, T2)> const&, f, T3 const&, z, T0 const&, x,
    return Foldable_t<T0>::foldr(f, z, x);)

// foldr1 :: (a -> a -> a) -> t a -> a
DEFINE_FUNCTION_2(4, fold1_type<T0>, foldr1, function_t<T3(T1, T2)> const&, f, T0 const&, x,
    return Foldable_t<T0>::foldr1(f, x);)

_FUNCPROG_END
