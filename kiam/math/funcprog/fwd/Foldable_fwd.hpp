#pragma once

#include "../funcprog_common.hpp"

_FUNCPROG_BEGIN

template<class _F>
struct _is_foldable : std::false_type {};

template<class _F>
bool constexpr _is_foldable_v = _is_foldable<_F>::value;

template<class F>
using is_foldable = _is_foldable<base_class_t<F> >;

template<class F>
bool constexpr is_foldable_v = is_foldable<F>::value;

template<class F, typename T = F>
using foldable_type = std::enable_if_t<is_foldable<F>::value, T>;

#define FOLDABLE_TYPE_(F, T) BOOST_IDENTITY_TYPE((foldable_type<F, T>))
#define FOLDABLE_TYPE(F, T) typename FOLDABLE_TYPE_(F, T)

// requires foldl, foldl1, foldr, foldr1
template<typename F>
struct Foldable;

template<typename T>
using Foldable_t = Foldable<base_class_t<T> >;

#define IMPLEMENT_FOLDABLE(_F) \
    template<> struct _is_foldable<_F> : std::true_type {}

/*
-- | Map each element of the structure to a monoid,
-- and combine the results.
foldMap :: Monoid m => (a -> m) -> t a -> m
{-# INLINE foldMap #-}
-- This INLINE allows more list functions to fuse. See Trac #9848.
foldMap f = foldr (mappend . f) mempty
*/
template<typename F, typename M, typename Arg>
using foldMap_type = std::enable_if_t<
    is_foldable<F>::value&& is_monoid_v<M> && is_same_as_v<value_type_t<F>, Arg>,
    M
>;

#define FOLDMAP_TYPE_(F, M, Arg) BOOST_IDENTITY_TYPE((foldMap_type<F, M, Arg>))
#define FOLDMAP_TYPE(F, M, Arg) typename FOLDMAP_TYPE_(F, M, Arg)

DECLARE_FUNCTION_2(3, FOLDMAP_TYPE(T0, T1, T2), foldMap, function_t<T1(T2)> const&, T0 const&);

    /*
    -- | Combine the elements of a structure using a monoid.
    fold :: Monoid m => t m -> m
    fold = foldMap id
    */
    template<typename F>
using fold_type = std::enable_if_t<
    is_foldable<F>::value&& is_monoid_v<value_type_t<F> >,
    value_type_t<F>
>;

// fold :: Monoid m => t m -> m
// fold = foldMap id
template<typename T>
constexpr fold_type<T> fold(T const& v);

template<typename FO, typename A, typename B, typename Ret>
using foldlr_type = std::enable_if_t<
    is_foldable_v<FO> && is_same_as_v<value_type_t<FO>, A> && is_same_as_v<Ret, B>,
    Ret
>;

#define FOLDLR_TYPE_(FO, A, B, Ret) BOOST_IDENTITY_TYPE((foldlr_type<FO, A, B, Ret>))
#define FOLDLR_TYPE(FO, A, B, Ret) typename FOLDLR_TYPE_(FO, A, B, Ret)

template<typename FO>
using fold1_type = foldable_type<FO, value_type_t<FO> >;

DECLARE_FUNCTION_3(4, FOLDLR_TYPE(T0, T1, T2, T3), foldl, function_t<T3(T2, T1)> const&, T3 const&, T0 const&);

// foldl1 :: (a -> a -> a) -> t a -> a
DECLARE_FUNCTION_2(4, fold1_type<T0>, foldl1, function_t<T3(T1, T2)> const&, T0 const&);

// foldr :: (a -> b -> b) -> b -> t a -> b
DECLARE_FUNCTION_3(4, FOLDLR_TYPE(T0, T1, T2, T3), foldr, function_t<T3(T1, T2)> const&, T3 const&, T0 const&);

// foldr1 :: (a -> a -> a) -> t a -> a
DECLARE_FUNCTION_2(4, fold1_type<T0>, foldr1, function_t<T3(T1, T2)> const&, T0 const&);

template<typename F>
constexpr foldable_type<F, bool> null(F const&);

template<typename F>
constexpr foldable_type<F, int> length(F const&);

template<typename F>
constexpr foldable_type<F, List<value_type_t<F> > > toList(F const& f);

template<typename F>
constexpr foldable_type<F, bool> and_(F const& f);

template<typename F>
constexpr foldable_type<F, bool> or_(F const& f);

template<typename F, typename Arg>
constexpr foldable_type<F, bool> any(function_t<bool(Arg)> const& p, F const& f);

template<typename F, typename Arg>
constexpr auto _any(function_t<bool(Arg)> const& p)
{
    static_assert(is_foldable_v<F>, "Should be Foldable");
    return _([p](F const& f) { return any(p, f); });
}


template<typename F, typename Arg>
constexpr foldable_type<F, bool> all(function_t<bool(Arg)> const& p, F const& f);

template<typename F, typename Arg>
constexpr auto _all(function_t<bool(Arg)> const& p)
{
    static_assert(is_foldable_v<F>, "Should be Foldable");
    return _([p](F const& f) { return all(p, f); });
}

template<typename F>
constexpr foldable_type<F, value_type_t<F> > maximum(F const& f);

template<typename F>
constexpr foldable_type<F, value_type_t<F> > minimum(F const& f);

template<typename F>
constexpr foldable_type<F, value_type_t<F> > sum(F const& f);

template<typename F>
constexpr foldable_type<F, value_type_t<F> > product(F const& f);

template<typename F, typename Arg1, typename Arg2>
constexpr foldable_type<F, value_type_t<F> >
maximumBy(function_t<Ordering(Arg1, Arg2)> const& cmp, F const& f);

template<typename F, typename Arg1, typename Arg2>
constexpr auto _maximumBy(function_t<Ordering(Arg1, Arg2)> const& cmp)
{
    static_assert(is_foldable_v<F>, "Should be Foldable");
    return _([cmp](F const& f) { return maximumBy(cmp, f); });
}

template<typename F, typename Arg1, typename Arg2>
constexpr foldable_type<F, value_type_t<F> >
minimumBy(function_t<Ordering(Arg1, Arg2)> const& cmp, F const& f);

template<typename F, typename Arg1, typename Arg2>
constexpr auto _minimumBy(function_t<Ordering(Arg1, Arg2)> const& cmp)
{
    static_assert(is_foldable_v<F>, "Should be Foldable");
    return _([cmp](F const& f) { return minimumBy(cmp, f); });
}

template<typename F>
constexpr foldable_type<F, bool> elem(value_type_t<F> const& a, F const& f);

template<typename F>
constexpr auto _elem(value_type_t<F> const& a)
{
    static_assert(is_foldable_v<F>, "Should be Foldable");
    return _([a](F const& f) { return elem(a, f); });
}

template<typename F>
constexpr foldable_type<F, bool>
notElem(value_type_t<F> const& a, F const& f);

template<typename F>
constexpr auto _notElem(value_type_t<F> const& a)
{
    static_assert(is_foldable_v<F>, "Should be Foldable");
    return _([a](F const& f) { return notElem(a, f); });
}

template<typename A> struct Maybe;

template<typename F, typename Arg>
constexpr foldable_type<F, Maybe<value_type_t<F> > >
find(function_t<bool(Arg)> const& p, F const& f);

template<typename F, typename Arg>
constexpr foldable_type<F, function_t<Maybe<value_type_t<F> >(F const&)> >
_find(function_t<bool(Arg)> const& p)
{
    return _([p](F const& f) { return find(p, f); });
}

DECLARE_FUNCTION_3(4, FOLDABLE_TYPE(T0, T1), foldrM, function_t<T1(T2, T3)> const&, value_type_t<T1> const&, T0 const&);
DECLARE_FUNCTION_3(4, FOLDABLE_TYPE(T0, T1), foldlM, function_t<T1(T3, T2)> const&, value_type_t<T1> const&, T0 const&);

DECLARE_FUNCTION_2(3, FOLDABLE_TYPE(T0, TYPEOF_T(T1, None)), traverse_, function_t<T1(T2)> const&, T0 const&);
DECLARE_FUNCTION_2(3, FOLDABLE_TYPE(T0, TYPEOF_T(T1, None)), for_, T0 const&, function_t<T1(T2)> const&);

DECLARE_FUNCTION_2(3, FOLDABLE_TYPE(T0, TYPEOF_T(T1, None)), mapM_, function_t<T1(T2)> const&, T0 const&);
DECLARE_FUNCTION_2(3, FOLDABLE_TYPE(T0, TYPEOF_T(T1, None)), forM_, T0 const&, function_t<T1(T2)> const&);

template<typename F>
foldable_type<F, typeof_t<value_type_t<F>, None> > sequenceA_(F const& xs);

template<typename F>
foldable_type<F, typeof_t<value_type_t<F>, None> > sequence_(F const& xs);

template<typename F>
foldable_type<F, value_type_t<F> > asum(F const& f);

template<typename F>
foldable_type<F, value_type_t<F> > msum(F const& f);

template<typename F>
foldable_type<F, value_type_t<F> > concat(F const& xs);

DECLARE_FUNCTION_2(3, FOLDABLE_TYPE(T0, List<T1>), concatMap, function_t<List<T1>(T2)> const&, T0 const&);

_FUNCPROG_END
