#pragma once

#include "fwd/Foldable_fwd.hpp"

_FUNCPROG_BEGIN

template<typename _F>
struct _Foldable // Default implementation of some functions
{
    // Combine the elements of a structure using a monoid.
    // fold :: Monoid m => t m -> m
    template<typename M>
    constexpr monoid_type<M> fold(typeof_t<_F, M> const& x);

    // Map each element of the structure to a monoid, and combine the results.
    // foldMap :: Monoid m => (a -> m) -> t a -> m
    template<typename M, typename Arg>
    static constexpr monoid_type<M> foldMap(function_t<M(Arg)> const& f, typeof_dt<_F, Arg> const& x);

    // List of elements of a structure, from left to right.
    // toList :: t a -> [a]
    template<typename A>
    static constexpr List<A> toList(typeof_t<_F, A> const& t);

    // Test whether the structure is empty. The default implementation is
    // optimized for structures that are similar to cons-lists, because there
    // is no general way to do better.
    // null :: t a -> Bool
    template<typename A>
    static constexpr bool null(typeof_t<_F, A> const& x);

    // Returns the size/length of a finite structure as an 'Int'.  The
    // default implementation is optimized for structures that are similar to
    // cons-lists, because there is no general way to do better.
    // length :: t a -> Int
    template<typename A>
    static constexpr int length(typeof_t<_F, A> const& x);

    // Does the element occur in the structure?
    // elem :: Eq a => a -> t a -> Bool
    template<typename A>
    static constexpr bool elem(A const& a, typeof_t<_F, A> const& f);

    // The largest element of a non-empty structure.
    // maximum :: forall a . Ord a => t a -> a
    template<typename A>
    static constexpr A maximum(typeof_t<_F, A> const& f);

    // The least element of a non-empty structure.
    // minimum :: forall a . Ord a => t a -> a
    template<typename A>
    static constexpr A minimum(typeof_t<_F, A> const& f);

    // The 'sum' function computes the sum of the numbers of a structure.
    // sum :: Num a => t a -> a
    template<typename A>
    static constexpr A sum(typeof_t<_F, A> const& f);

    // The 'product' function computes the product of the numbers of a structure.
    // product :: Num a => t a -> a
    template<typename A>
    static constexpr A product(typeof_t<_F, A> const& f);
};

#define DECLARE_FOLDABLE_CLASS(F) \
    /* foldl :: (b -> a -> b) -> b -> t a -> b */ \
    template<typename Ret, typename A, typename B> \
    static constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret> \
    foldl(function_t<Ret(B, A)> const& f, Ret const& z, F<fdecay<A> > const& x); \
    \
    /* foldl1 :: (a -> a -> a) -> t a -> a */ \
    template<typename A, typename Arg1, typename Arg2> \
    static constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A> \
    foldl1(function_t<A(Arg1, Arg2)> const& f, F<A> const& x); \
    \
    /* foldr :: (a -> b -> b) -> b -> t a -> b */ \
    template<typename Ret, typename A, typename B> \
    static constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret> \
    foldr(function_t<Ret(A, B)> const& f, Ret const& z, F<fdecay<A> > const& x); \
    \
    /* foldr1 :: (a -> a -> a) -> t a -> a */ \
    template<typename A, typename Arg1, typename Arg2> \
    static constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A> \
    foldr1(function_t<A(Arg1, Arg2)> const& f, F<A> const& x); \

_FUNCPROG_END
