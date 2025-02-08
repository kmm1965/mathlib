#pragma once

#include "../MonadPlus.hpp"
#include "../Monad.hpp"
#include "../Alternative.hpp"

_FUNCPROG_BEGIN

template<typename _MP>
template<typename A>
constexpr alternative_type<typeof_t<_MP, A> >
_MonadPlus<_MP>::mzero(){
    return Alternative<_MP>::template empty<A>();
}

template<typename _MP>
template<typename A>
constexpr alternative_type<typeof_t<_MP, A> >
_MonadPlus<_MP>::mplus(typeof_t<_MP, A> const& x, typeof_t<_MP, A> const& y){
    return Alternative<_MP>::alt_op(x, y);
}

FUNCTION_TEMPLATE(2) constexpr MPLUS_TYPE(T0, T1) mplus(T0 const& x, T1 const& y) {
    return MonadPlus_t<T1>::mplus(x, y);
}

// Translate a list to an arbitrary 'MonadPlus' type.
// This function generalizes the 'listToMaybe' function.
//mfromList :: MonadPlus m => [a] -> m a
//mfromList = Monad.msum . map return
template<typename _MP, typename A>
constexpr _monad_plus_type<_MP, typeof_t<_MP, A> >
mfromList(List<A> const& l){
    return msum(fmap(_(Monad<_MP>::template mreturn<A>), l));
}

// Translate maybe to an arbitrary 'MonadPlus' type.
// This function generalizes the 'maybeToList' function.
//mfromMaybe :: MonadPlus m => Maybe a -> m a
//mfromMaybe = maybe mzero return
template<typename _MP, typename A>
constexpr _monad_plus_type<_MP, typeof_t<_MP, A> >
mfromMaybe(Maybe<A> const& x){
    return maybe(MonadPlus<_MP>::template mzero<A>(), _(Monad<_MP>::template mreturn<A>), x);
}

// Fold a value into an arbitrary 'MonadPlus' type.
// This function generalizes the 'toList' function.
//mfold :: (MonadPlus m, Foldable t) => t a -> m a
//mfold = mfromList . Foldable.toList
template<typename _MP, typename F>
constexpr monad_plus_type<typeof_t<_MP, value_type_t<F> > >
mfold(F const& f){
    static_assert(is_foldable_v<F>, "Should be a Foldable");
    return mfromList<_MP>(toList(f));
}

// Convert a partial function to a function returning an arbitrary
// 'MonadPlus' type.
//mreturn :: MonadPlus m => (a -> Maybe b) -> a -> m b
//mreturn f = mfromMaybe . f
FUNCTION_TEMPLATE(3) constexpr _MONAD_PLUS_TYPE(T0, TYPEOF_T(T0, T1)) mreturn(function_t<Maybe<T1>(T2)> const& f, fdecay<T2> const& x) {
    return mfromMaybe<T0>(f(x));
}

// -----------------------------------------------------------------------------
// | Direct 'MonadPlus' equivalent of 'Data.List.filter'.
//
// ==== __Examples__
//
// The 'Data.List.filter' function is just 'mfilter' specialized to
// the list monad:
//
// 'Data.List.filter' = ( 'mfilter' :: (a -> Bool) -> [a] -> [a] )
//
// An example using 'mfilter' with the 'Maybe' monad:
//
// >>> mfilter odd (Just 1)
// Just 1
// >>> mfilter odd (Just 2)
// Nothing
//mfilter :: (MonadPlus m) => (a -> Bool) -> m a -> m a
//mfilter p ma = do
//  a <- ma
//  if p a then return a else mzero
FUNCTION_TEMPLATE(2) constexpr monad_plus_type<T0> mfilter(function_t<bool(T1)> const& p, T0 const& ma)
{
    static_assert(is_same_as_v<T1, value_type_t<T0> >, "Should be the same");
    return _do(a, ma, return p(a) ? Monad_t<T0>::mreturn(a) : MonadPlus_t<T0>::template mzero<value_type_t<T0> >(););
}

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
//template<typename MP, typename Arg>
FUNCTION_TEMPLATE(2) constexpr MONAD_PLUS_TYPE(T0, PAIR_T(T0, T0)) mpartition(function_t<bool(T1)> const& p, T0 const& ma)
{
    static_assert(is_same_as_v<T1, value_type_t<T0> >, "Should be the same");
    return PAIR_T(T0, T0)(mfilter(p, ma), mfilter(_(not__) & p, ma));
}

// Pass through @Just@ elements.
// This function generalizes the 'catMaybes' function.
//mcatMaybes :: MonadPlus m => m (Maybe a) -> m a
//mcatMaybes = (>>= mfromMaybe)
template<typename MP>
using mcatMaybes_type = monad_plus_type<MP, typeof_t<MP, value_type_t<value_type_t<MP> > > >;

template<typename MP>
constexpr mcatMaybes_type<MP> mcatMaybes(MP const& xs){
    static_assert(is_maybe_v<value_type_t<MP> >, "Should be a Maybe");
    return xs >>= _(mfromMaybe<base_class_t<MP>, value_type_t<value_type_t<MP> > >);
}

// Join list elements together.
// This function generalizes the 'catMaybes' function.
//mscatter :: MonadPlus m => m [b] -> m b
//mscatter = (>>= mfromList)
template<typename MP>
constexpr mscatter_type<MP> mscatter(MP const& xs){
    static_assert(is_list_v<value_type_t<MP> >, "Should be a List");
    return xs >>= _(mfromList<base_class_t<MP>, value_type_t<value_type_t<MP> > >);
}

// Join foldable elements together.
// This function generalizes the 'catMaybes' function.
//mscatter' :: (MonadPlus m, Foldable t) => m (t b) -> m b
//mscatter' = (>>= mfold)
template<typename MP>
constexpr mscatter_type<MP> mscatter_(MP const& xs){
    static_assert(is_foldable_v<value_type_t<MP> >, "Should be a Foldable");
    return xs >>= _(mfold<base_class_t<MP>, value_type_t<MP> >);
}

// Pass through @Left@ elements.
// This function generalizes the 'lefts' function.
//mlefts :: MonadPlus m => m (Either a b) -> m a
//mlefts = mcatMaybes . liftM l
//    where
//        l (Left a)  = Just a
//        l (Right a) = Nothing
template<typename MP>
constexpr mlefts_type<MP> mlefts(MP const& xs){
    static_assert(is_either_v<value_type_t<MP> >, "Should be an Either");
    using A = typename value_type_t<MP>::left_type;
    auto const l = _([](value_type_t<MP> const& v){
        return v.index() == Left_ ? Just(*v.left()) : Nothing<A>();
    });
    return mcatMaybes(liftM(l, xs));
}

// Pass through @Right@ elements.
// This function generalizes the 'rights' function.
//mrights :: MonadPlus m => m (Either a b) -> m b
//mrights = mcatMaybes . liftM r
//    where
//        r (Left a)  = Nothing
//        r (Right a) = Just a
template<typename MP>
constexpr mrights_type<MP> mrights(MP const& xs){
    static_assert(is_either_v<value_type_t<MP> >, "Should be an Either");
    using B = typename value_type_t<MP>::right_type;
    auto const r = _([](value_type_t<MP> const& v){
        return v.index() == Right_ ? Just(*v.right()) : Nothing<B>();
    });
    return mcatMaybes(liftM(r, xs));
}

// Separate @Left@ and @Right@ elements.
// This function generalizes the 'partitionEithers' function.
//mpartitionEithers :: MonadPlus m => m (Either a b) -> (m a, m b)
//mpartitionEithers a = (mlefts a, mrights a)

template<typename MP>
constexpr mpartitionEithers_type<MP> mpartitionEithers(MP const& xs){
    static_assert(is_either_v<value_type_t<MP> >, "Should be an Either");
    return std::make_pair(mlefts(xs), mrights(xs));
}

// Modify or discard a value.
// This function generalizes the 'mapMaybe' function.
//mmapMaybe :: MonadPlus m => (a -> Maybe b) -> m a -> m b
//mmapMaybe f = mcatMaybes . liftM f
FUNCTION_TEMPLATE(3) constexpr MONAD_PLUS_TYPE(T0, TYPEOF_T(T0, T1)) mmapMaybe(function_t<Maybe<T1>(T2)> const& f, T0 const& xs)
{
    static_assert(is_same_as_v<T2, value_type_t<T0> >, "Should be the same");
    return mcatMaybes(liftM(f, xs));
}

// Modify, discard or spawn values.
// This function generalizes the 'concatMap' function.
//mconcatMap :: MonadPlus m => (a -> [b]) -> m a -> m b
//mconcatMap f = mscatter . liftM f
FUNCTION_TEMPLATE(3) constexpr MONAD_PLUS_TYPE(T0, TYPEOF_T(T0, T1)) mconcatMap(function_t<List<T1>(T2)> const& f, T0 const& xs)
{
    static_assert(is_same_as_v<T2, value_type_t<T0> >, "Should be the same");
    return mscatter(liftM(f, xs));
}

// Modify, discard or spawn values.
// This function generalizes the 'concatMap' function.
//mconcatMap' :: (MonadPlus m, Foldable t) => (a -> t b) -> m a -> m b
//mconcatMap' f = mscatter' . liftM f
FUNCTION_TEMPLATE(3) constexpr MCONCATMAP__TYPE(T0, T1, T2) mconcatMap_(function_t<T1(T2)> const& f, T0 const& xs)
{
    static_assert(is_foldable_v<T1>, "Should be a Foldable");
    static_assert(is_same_as_v<T2, value_type_t<T0> >, "Should be the same");
    return mscatter_(liftM(f, xs));
}

// Convert a predicate to a partial function.
//partial :: (a -> Bool) -> a -> Maybe a
//partial p x = if p x then Just x else Nothing
FUNCTION_TEMPLATE(1) constexpr Maybe<fdecay<T0> > partial(function_t<bool(T0)> const& p, fdecay<T0> const& x) {
    return p(x) ? Just(x) : Nothing<fdecay<T0> >();
}

// Convert a partial function to a predicate.
//predicate :: (a -> Maybe a) -> a -> Bool
//predicate f x = case f x of
//    Just _  -> True
//    Nothing -> False
FUNCTION_TEMPLATE(2) constexpr PREDICATE_TYPE(T0, T1) predicate(function_t<Maybe<T0>(T1)> const& f, T0 const& x) {
    return (bool)f(x);
}

// Convert a total function to a partial function.
//always :: (a -> b) -> a -> Maybe b
//always f = Just . f
FUNCTION_TEMPLATE(2) constexpr Maybe<T0> always(function_t<T0(T1)> const& f, fdecay<T1> const& x) {
    return Just(f(x));
}

// Make a partial function that always rejects its input.
//never :: a -> Maybe c
//never = const Nothing
template<typename A>
constexpr Maybe<A> never(A const& x){
    return Nothing<A>();
}

_FUNCPROG_END
