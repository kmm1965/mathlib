#pragma once

#include "../funcprog_common.hpp"

_FUNCPROG_BEGIN

template<class _MP>
struct _is_monad_plus : std::false_type {};

template<class _MP>
bool constexpr _is_monad_plus_v = _is_monad_plus<_MP>::value;

template<class _MP, typename T>
using _monad_plus_type = std::enable_if_t<_is_monad_plus_v<_MP>, T>;

#define _MONAD_PLUS_TYPE_(_MP, T) BOOST_IDENTITY_TYPE((_monad_plus_type<_MP, T>))
#define _MONAD_PLUS_TYPE(_MP, T) typename _MONAD_PLUS_TYPE_(_MP, T)

template<class MP>
using is_monad_plus = _is_monad_plus<base_class_t<MP> >;

template<class MP>
bool constexpr is_monad_plus_v = is_monad_plus<MP>::value;

template<class MP, typename T = MP>
using monad_plus_type = std::enable_if_t<is_monad_plus_v<MP>, T>;

#define MONAD_PLUS_TYPE_(MP, T) BOOST_IDENTITY_TYPE((monad_plus_type<MP, T>))
#define MONAD_PLUS_TYPE(MP, T) typename MONAD_PLUS_TYPE_(MP, T)

template<class _MP1, class _MP2>
struct _is_same_monad_plus : std::false_type {};

template<class _MP>
struct _is_same_monad_plus<_MP, _MP> : _is_monad_plus<_MP> {};

template<class _MP1, class _MP2>
bool constexpr _is_same_monad_plus_v = _is_same_monad_plus<_MP1, _MP2>::value;

template<class MP1, class MP2>
using is_same_monad_plus = _is_same_monad_plus<base_class_t<MP1>, base_class_t<MP2> >;

template<class MP1, class MP2>
bool constexpr is_same_monad_plus_v = is_same_monad_plus<MP1, MP2>::value;

template<class MP1, class MP2, typename T>
using same_monad_plus_type = std::enable_if_t<is_same_monad_plus<MP1, MP2>::value, T>;

#define SAME_MONAD_PLUS_TYPE_(MP1, MP2, T) BOOST_IDENTITY_TYPE((same_monad_plus_type<MP1, MP2, T>))
#define SAME_MONAD_PLUS_TYPE(MP1, MP2, T) typename SAME_MONAD_PLUS_TYPE_(MP1, MP2, T)

// Requires mzero (default empty), mplus (default |)
template<typename _MP>
struct MonadPlus;

template<typename A>
using MonadPlus_t = MonadPlus<base_class_t<A> >;

#define IMPLEMENT_MONADPLUS(_MP) \
    template<> struct _is_monad_plus<_MP> : std::true_type {}

template<typename MP1, typename MP2>
using mplus_type = same_monad_plus_type<MP1, MP2, MP2>;

#define MPLUS_TYPE_(MP1, MP2) BOOST_IDENTITY_TYPE((mplus_type<MP1, MP2>))
#define MPLUS_TYPE(MP1, MP2) typename MPLUS_TYPE_(MP1, MP2)

DECLARE_FUNCTION_2(2, MPLUS_TYPE(T0, T1), mplus, T0 const&, T1 const&);

template<typename A> struct List;

//-- Translate a list to an arbitrary 'MonadPlus' type.
//-- This function generalizes the 'listToMaybe' function.
//mfromList :: MonadPlus m => [a] -> m a
template<typename _MP, typename A>
constexpr _monad_plus_type<_MP, typeof_t<_MP, A> > mfromList(List<A> const& l);

template<typename A> struct Maybe;

//-- Translate maybe to an arbitrary 'MonadPlus' type.
//-- This function generalizes the 'maybeToList' function.
//mfromMaybe :: MonadPlus m => Maybe a -> m a
template<typename _MP, typename A>
constexpr _monad_plus_type<_MP, typeof_t<_MP, A> > mfromMaybe(Maybe<A> const& x);

//-- Fold a value into an arbitrary 'MonadPlus' type.
//-- This function generalizes the 'toList' function.
//mfold :: (MonadPlus m, Foldable t) => t a -> m a
template<typename _MP, typename F>
constexpr monad_plus_type<typeof_t<_MP, value_type_t<F> > > mfold(F const& f);

//-- Convert a partial function to a function returning an arbitrary
//-- 'MonadPlus' type.
//mreturn :: MonadPlus m => (a -> Maybe b) -> a -> m b
//template<typename _MP, typename B, typename Arg>
DECLARE_FUNCTION_2(3, _MONAD_PLUS_TYPE(T0, TYPEOF_T(T0, T1)), mreturn, function_t<Maybe<T1>(T2)> const&, fdecay<T2> const&);

// Direct 'MonadPlus' equivalent of 'Data.List.filter'.
//mfilter :: (MonadPlus m) => (a -> Bool) -> m a -> m a
DECLARE_FUNCTION_2(2, monad_plus_type<T0>, mfilter, function_t<bool(T1)> const&, T0 const&);

// The 'partition' function takes a predicate a list and returns
// the pair of lists of elements which do and do not satisfy the
// predicate, respectively; i.e.,
//
// > partition p xs == (filter p xs, filter (not . p) xs)
//
// This function generalizes the 'partition' function.
//
//mpartition :: MonadPlus m => (a -> Bool) -> m a -> (m a, m a)
//mpartition p a = (mfilter p a, mfilter (not . p) a)
DECLARE_FUNCTION_2(2, MONAD_PLUS_TYPE(T0, PAIR_T(T0, T0)), mpartition, function_t<bool(T1)> const&, T0 const&);

// Pass through @Just@ elements.
// This function generalizes the 'catMaybes' function.
//mcatMaybes :: MonadPlus m => m (Maybe a) -> m a
template<typename MP>
using mcatMaybes_type = monad_plus_type<MP, typeof_t<MP, value_type_t<value_type_t<MP> > > >;

template<typename MP>
constexpr mcatMaybes_type<MP> mcatMaybes(MP const& xs);

// Join list elements together.
// This function generalizes the 'catMaybes' function.
//mscatter :: MonadPlus m => m [b] -> m b
template<typename MP>
using mscatter_type = mcatMaybes_type<MP>;

template<typename MP>
constexpr mscatter_type<MP> mscatter(MP const& xs);

// Join foldable elements together.
// This function generalizes the 'catMaybes' function.
//mscatter' :: (MonadPlus m, Foldable t) => m (t b) -> m b
template<typename MP>
constexpr mscatter_type<MP> mscatter_(MP const& xs);

// Pass through @Left@ elements.
// This function generalizes the 'lefts' function.
//mlefts :: MonadPlus m => m (Either a b) -> m a
template<typename MP>
using mlefts_type = monad_plus_type<typeof_t<MP, typename value_type_t<MP>::left_type> >;

template<typename MP>
constexpr mlefts_type<MP> mlefts(MP const& xs);

// Pass through @Right@ elements.
// This function generalizes the 'rights' function.
//mrights :: MonadPlus m => m (Either a b) -> m b
template<typename MP>
using mrights_type = monad_plus_type<typeof_t<MP, typename value_type_t<MP>::right_type> >;

template<typename MP>
constexpr mrights_type<MP> mrights(MP const& xs);

// Separate @Left@ and @Right@ elements.
// This function generalizes the 'partitionEithers' function.
//mpartitionEithers :: MonadPlus m => m (Either a b) -> (m a, m b)
template<typename MP>
using mpartitionEithers_type = monad_plus_type<MP,
    pair_t<
        typeof_t<MP, typename value_type_t<MP>::left_type>,
        typeof_t<MP, typename value_type_t<MP>::right_type>
    >
>;

template<typename MP>
constexpr mpartitionEithers_type<MP> mpartitionEithers(MP const& xs);

// Modify or discard a value.
// This function generalizes the 'mapMaybe' function.
//mmapMaybe :: MonadPlus m => (a -> Maybe b) -> m a -> m b
DECLARE_FUNCTION_2(3, MONAD_PLUS_TYPE(T0, TYPEOF_T(T0, T1)), mmapMaybe, function_t<Maybe<T1>(T2)> const&, T0 const&);

// Modify, discard or spawn values.
// This function generalizes the 'concatMap' function.
//mconcatMap :: MonadPlus m => (a -> [b]) -> m a -> m b
DECLARE_FUNCTION_2(3, MONAD_PLUS_TYPE(T0, TYPEOF_T(T0, T1)), mconcatMap, function_t<List<T1>(T2)> const&, T0 const&);

// Modify, discard or spawn values.
// This function generalizes the 'concatMap' function.
//mconcatMap' :: (MonadPlus m, Foldable t) => (a -> t b) -> m a -> m b
template<typename MP, typename F, typename Arg>
using mconcatMap__type = monad_plus_type<typeof_t<MP, value_type_t<F> > >;

#define MCONCATMAP__TYPE_(MP, F, Arg) BOOST_IDENTITY_TYPE((mconcatMap__type<MP, F, Arg>))
#define MCONCATMAP__TYPE(MP, F, Arg) typename MCONCATMAP__TYPE_(MP, F, Arg)

DECLARE_FUNCTION_2(3, MCONCATMAP__TYPE(T0, T1, T2), mconcatMap_, function_t<T1(T2)> const&, T0 const&);

// Convert a predicate to a partial function.
//partial :: (a -> Bool) -> a -> Maybe a
DECLARE_FUNCTION_2(1, Maybe<fdecay<T0> >, partial, function_t<bool(T0)> const&, fdecay<T0> const&);

// Convert a partial function to a predicate.
//predicate :: (a -> Maybe a) -> a -> Bool
template<typename A, typename Arg>
using predicate_type = std::enable_if_t<is_same_as_v<Arg, A>, bool>;

#define PREDICATE_TYPE_(A, Arg) BOOST_IDENTITY_TYPE((predicate_type<A, Arg>))
#define PREDICATE_TYPE(A, Arg) typename PREDICATE_TYPE_(A, Arg)

DECLARE_FUNCTION_2(2, PREDICATE_TYPE(T0, T1), predicate, function_t<Maybe<T0>(T1)> const&, T0 const&);

// Convert a total function to a partial function.
//always :: (a -> b) -> a -> Maybe b
DECLARE_FUNCTION_2(2, Maybe<T0>, always, function_t<T0(T1)> const&, fdecay<T1> const&);

// Make a partial function that always rejects its input.
//never :: a -> Maybe c
template<typename A>
constexpr Maybe<A> never(A const& x);

_FUNCPROG_END
