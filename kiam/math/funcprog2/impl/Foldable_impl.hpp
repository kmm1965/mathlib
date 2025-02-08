#pragma once

#include "../Foldable.hpp"
#include "../fwd/List_fwd.hpp"
#include "Foldable.Data.hpp"

_FUNCPROG2_BEGIN

// Combine the elements of a structure using a monoid.
// fold :: Monoid m => t m -> m
// fold = foldMap id
template<typename _F>
template<typename M>
constexpr monoid_type<M> _Foldable<_F>::fold(typeof_t<_F, M> const& x){
    return Foldable<_F>::foldMap(_(id<M>), x);
}

// Map each element of the structure to a monoid, and combine the results.
// foldMap :: Monoid m => (a -> m) -> t a -> m
// foldMap f = foldr (mappend . f) mempty
template<typename _F>
template<typename M, typename Arg>
constexpr monoid_type<M> _Foldable<_F>::foldMap(function_t<M(Arg)> const& f, typeof_dt<_F, Arg> const& x){
    return Foldable<_F>::foldr(_(Monoid_t<M>::template mappend<M>) & f, M::mempty(), x);
}

// List of elements of a structure, from left to right.
// toList :: t a -> [a]
// toList t = build (\ c n -> foldr c n t)
template<typename _F>
template<typename A>
constexpr List<A> _Foldable<_F>::toList(typeof_t<_F, A> const& t){
    return build(_([t](function_t<List<A>(A const&, List<A> const&)> const& c, List<A> const& n){
        return Foldable<_F>::foldr(c, n, t); }));
}

// Test whether the structure is empty. The default implementation is
// optimized for structures that are similar to cons-lists, because there
// is no general way to do better.
// null :: t a -> Bool
// null = foldr (\_ _ -> False) True
template<typename _F>
template<typename A>
constexpr bool _Foldable<_F>::null(typeof_t<_F, A> const& x){
    return Foldable<_F>::foldr(_([](A const&, bool){ return false; }), true, x);
}

// Returns the size/length of a finite structure as an 'Int'.  The
// default implementation is optimized for structures that are similar to
// cons-lists, because there is no general way to do better.
// length :: t a -> Int
// length = foldl' (\c _ -> c+1) 0
template<typename _F>
template<typename A>
constexpr int _Foldable<_F>::length(typeof_t<_F, A> const& x){
    return Foldable<_F>::foldl(_([](int c, A const&){ return c + 1; }), 0, x);
}

// Does the element occur in the structure?
// elem :: Eq a => a -> t a -> Bool
// elem = any . (==)
template<typename _F>
template<typename A>
constexpr bool _Foldable<_F>::elem(A const& a, typeof_t<_F, A> const& f){
    return (_(any<typeof_t<_F, A>, A const&>) & _(eq<A>))(a, f);
}

// The largest element of a non-empty structure.
// maximum :: forall a . Ord a => t a -> a
// maximum = fromMaybe (errorWithoutStackTrace "maximum: empty structure") .
//    getMax . foldMap (Max #. (Just :: a -> Maybe a))
template<typename _F>
template<typename A>
constexpr A _Foldable<_F>::maximum(typeof_t<_F, A> const& f){
    using F = typeof_t<_F, A>;
    return (_fromMaybe<A>(_errorWithoutStackTrace<A>("maximum: empty structure")) &
        _(getMax<A>) & _foldMap<F>(_(Max_<A>) & _(Just<A>)))(f);
}

// The least element of a non-empty structure.
// minimum :: forall a . Ord a => t a -> a
// minimum = fromMaybe (errorWithoutStackTrace "minimum: empty structure") .
//    getMin . foldMap (Min #. (Just :: a -> Maybe a))
template<typename _F>
template<typename A>
constexpr A _Foldable<_F>::minimum(typeof_t<_F, A> const& f){
    using F = typeof_t<_F, A>;
    return (_fromMaybe<A>(_errorWithoutStackTrace<A>("minimum: empty structure")) &
        _(getMin<A>) & _foldMap<F>(_(Min_<A>) & _(Just<A>)))(f);
}

// The 'sum' function computes the sum of the numbers of a structure.
// sum :: Num a => t a -> a
// sum = getSum #. foldMap Sum
template<typename _F>
template<typename A>
constexpr A _Foldable<_F>::sum(typeof_t<_F, A> const& f){
    using F = typeof_t<_F, A>;
    return (_(getSum<A>) & _foldMap<F>(_(Sum_<A>)))(f);
}

// The 'product' function computes the product of the numbers of a structure.
// product :: Num a => t a -> a
// product = getProduct #. foldMap Product
template<typename _F>
template<typename A>
constexpr A _Foldable<_F>::product(typeof_t<_F, A> const& f){
    using F = typeof_t<_F, A>;
    return (_(getProduct<A>) & _foldMap<F>(_(Product_<A>)))(f);
}

/*
-- | Map each element of the structure to a monoid,
-- and combine the results.
foldMap :: Monoid m => (a -> m) -> t a -> m
{-# INLINE foldMap #-}
-- This INLINE allows more list functions to fuse. See Trac #9848.
foldMap f = foldr (mappend . f) mempty
*/
FUNCTION_TEMPLATE(3) constexpr FOLDMAP_TYPE(T0, T1, T2) foldMap(function_t<T1(T2)> const& f, T0 const& v) {
    return Foldable_t<T0>::foldMap(f, v);
}

/*
-- | Combine the elements of a structure using a monoid.
fold :: Monoid m => t m -> m
fold = foldMap id
*/
// fold :: Monoid m => t m -> m
// fold = foldMap id
template<typename T>
constexpr fold_type<T> fold(T const& v){
    return foldMap(_(id<value_type_t<T> >), v);
}

FUNCTION_TEMPLATE(4) constexpr FOLDLR_TYPE(T0, T1, T2, T3) foldl(function_t<T3(T2, T1)> const& f, T3 const& z, T0 const& x) {
    return Foldable_t<T0>::foldl(f, z, x);
}

// foldl1 :: (a -> a -> a) -> t a -> a
FUNCTION_TEMPLATE(4) constexpr fold1_type<T0> foldl1(function_t<T3(T1, T2)> const& f, T0 const& x) {
    return Foldable_t<T0>::foldl1(f, x);
}

// foldr :: (a -> b -> b) -> b -> t a -> b
FUNCTION_TEMPLATE(4) constexpr FOLDLR_TYPE(T0, T1, T2, T3) foldr(function_t<T3(T1, T2)> const& f, T3 const& z, T0 const& x) {
    return Foldable_t<T0>::foldr(f, z, x);
}

// foldr1 :: (a -> a -> a) -> t a -> a
FUNCTION_TEMPLATE(4) constexpr fold1_type<T0> foldr1(function_t<T3(T1, T2)> const& f, T0 const& x) {
    return Foldable_t<T0>::foldr1(f, x);
}

_FUNCPROG2_END

#include "Foldable.Extra.hpp"
