#pragma once

_FUNCPROG_BEGIN

template<typename F>
constexpr foldable_type<F, bool> null(F const& f){
    return Foldable_t<F>::null(f);
}

template<typename F>
constexpr foldable_type<F, int> length(F const& f){
    return Foldable_t<F>::length(f);
}

template<typename F>
constexpr foldable_type<F, List<value_type_t<F> > > toList(F const& f){
    return Foldable_t<F>::template toList<value_type_t<F> >(f);
}

//-- | 'and' returns the conjunction of a container of Bools.For the
//-- result to be 'True', the container must be finite; 'False', however,
//--results from a 'False' value finitely far from the left end.
//and ::Foldable t = > t Bool->Bool
//and = getAll #.foldMap All
template<typename F>
constexpr foldable_type<F, bool> and_(F const& f){
    return (_(getAll) & _foldMap<F>(_(All_)))(f);
}

//-- | 'or' returns the disjunction of a container of Bools.For the
//-- result to be 'False', the container must be finite; 'True', however,
//--results from a 'True' value finitely far from the left end.
//or ::Foldable t = > t Bool->Bool
//or = getAny #.foldMap Any
template<typename F>
constexpr foldable_type<F, bool> or_(F const& f){
    //return Foldable_t<F>::foldMap(_(Any_), f).get();
    return (_(getAny) & _foldMap<F>(_(Any_)))(f);
}

//-- | Determines whether any element of the structure satisfies the predicate.
//any::Foldable t = > (a->Bool)->t a->Bool
//any p = getAny #.foldMap(Any #.p)
template<typename F, typename Arg>
constexpr foldable_type<F, bool> any(function_t<bool(Arg)> const& p, F const& f){
    static_assert(is_same_as_v<Arg, value_type_t<F> >, "Should be the same");
    return (_(getAny) & _foldMap<F>(_(Any_) & p))(f);
}

//-- | Determines whether all elements of the structure satisfy the predicate.
//all::Foldable t = > (a->Bool)->t a->Bool
//all p = getAll #.foldMap(All #.p)
template<typename F, typename Arg>
constexpr foldable_type<F, bool> all(function_t<bool(Arg)> const& p, F const& f){
    static_assert(is_same_as_v<Arg, value_type_t<F> >, "Should be the same");
    return (_(getAll) & _foldMap<F>(_(All_) & p))(f);
}

template<typename F>
constexpr foldable_type<F, value_type_t<F> > maximum(F const& f){
    return Foldable_t<F>::maximum(f);
}

template<typename F>
constexpr foldable_type<F, value_type_t<F> > minimum(F const& f){
    return Foldable_t<F>::minimum(f);
}

//-- | The 'sum' function computes the sum of the numbers of a structure.
//  sum :: Num a => t a -> a
//  sum = getSum #. foldMap Sum
template<typename F>
constexpr foldable_type<F, value_type_t<F> > sum(F const& f){
    return Foldable_t<F>::sum(f);
}

//-- | The 'product' function computes the product of the numbers of a structure.
//  product :: Num a => t a -> a
//  product = getProduct #. foldMap Product
template<typename F>
constexpr foldable_type<F, value_type_t<F> > product(F const& f){
    return Foldable_t<F>::product(f);
}

//-- | The largest element of a non-empty structure with respect to the
//-- given comparison function.
//
//-- See Note [maximumBy/minimumBy space usage]
//maximumBy :: Foldable t => (a -> a -> Ordering) -> t a -> a
//maximumBy cmp = foldl1 max'
//  where max' x y = case cmp x y of
//                        GT -> x
//                        _  -> y
template<typename F, typename Arg1, typename Arg2>
constexpr foldable_type<F, value_type_t<F> >
maximumBy(function_t<Ordering(Arg1, Arg2)> const& cmp, F const& f){
    using A = value_type_t<F>;
    static_assert(is_same_as_v<Arg1, A>, "Should be the same");
    static_assert(is_same_as_v<Arg2, A>, "Should be the same");
    function_t<A(A const&, A const&)> max_ = [&cmp](A const& x, A const& y){
        return cmp(x, y) == GT ? x : y;
    };
    return foldl1(max_, f);
}

//-- | The least element of a non-empty structure with respect to the
//-- given comparison function.
//
//-- See Note [maximumBy/minimumBy space usage]
//minimumBy :: Foldable t => (a -> a -> Ordering) -> t a -> a
//minimumBy cmp = foldl1 min'
//  where min' x y = case cmp x y of
//                        GT -> y
//                        _  -> x
template<typename F, typename Arg1, typename Arg2>
constexpr foldable_type<F, value_type_t<F> >
minimumBy(function_t<Ordering(Arg1, Arg2)> const& cmp, F const& f){
    using A = value_type_t<F>;
    static_assert(is_same_as_v<Arg1, A>, "Should be the same");
    static_assert(is_same_as_v<Arg2, A>, "Should be the same");
    function_t<A(A const&, A const&)> min_ = [&cmp](A const& x, A const& y){
        return cmp(x, y) == GT ? y : x;
    };
    return foldl1(min_, f);
}

//-- | Does the element occur in the structure?
//--
//elem :: Eq a => a -> t a -> Bool
template<typename F>
constexpr foldable_type<F, bool> elem(value_type_t<F> const& a, F const& f){
    return Foldable_t<F>::elem(a, f);
}

//-- | 'notElem' is the negation of 'elem'.
//notElem :: (Foldable t, Eq a) => a -> t a -> Bool
//notElem x = not . elem x
template<typename F>
constexpr foldable_type<F, bool>
notElem(value_type_t<F> const& a, F const& f){
    return (_(not__) & _(_elem<F>(a)))(f);
}

//-- | The 'find' function takes a predicate and a structure and returns
//-- the leftmost element of the structure matching the predicate, or
//-- 'Nothing' if there is no such element.
//find :: Foldable t => (a -> Bool) -> t a -> Maybe a
//find p = getFirst . foldMap (\ x -> First (if p x then Just x else Nothing))
template<typename F, typename Arg>
constexpr foldable_type<F, Maybe<value_type_t<F> > >
find(function_t<bool(Arg)> const& p, F const& f)
{
    using A = value_type_t<F>;
    static_assert(is_same_as_v<Arg, A>, "Should be the same");
    return (_(getFirst<A>) & _foldMap<F>(_([p](A const& x){ return First(p(x) ? Just(x) : Nothing<A>()); })))(f);
}

//-- | Monadic fold over the elements of a structure,
//-- associating to the right, i.e. from right to left.
//foldrM :: (Foldable t, Monad m) => (a -> b -> m b) -> b -> t a -> m b
//foldrM f z0 xs = foldl c return xs z0
//  -- See Note [List fusion and continuations in 'c']
//  where c k x z = f x z >>= k
FUNCTION_TEMPLATE(4) constexpr FOLDABLE_TYPE(T0, T1) foldrM(function_t<T1(T2, T3)> const& f, value_type_t<T1> const& z0, T0 const& xs)
{
    using F = T0; using M = T1;
    static_assert(is_monad_v<M>, "Should be a Monad");
    using A = value_type_t<F>;
    using B = value_type_t<M>;
    static_assert(is_same_as_v<T2, A>, "Should be the same");
    static_assert(is_same_as_v<T3, B>, "Should be the same");
    auto const c = _([f](function_t<T1(B const&)> const& k, A const& x) {
        return _([f, k, x](B const& z) {
            return f(x, z) >>= k;
        });
    });
    return Foldable_t<F>::foldl(c, _(Monad_t<M>::template mreturn<B>), xs)(z0);
}

//-- | Monadic fold over the elements of a structure,
//-- associating to the left, i.e. from left to right.
//foldlM :: (Foldable t, Monad m) => (b -> a -> m b) -> b -> t a -> m b
//foldlM f z0 xs = foldr c return xs z0
//  -- See Note [List fusion and continuations in 'c']
//  where c x k z = f z x >>= k
FUNCTION_TEMPLATE(4) constexpr FOLDABLE_TYPE(T0, T1) foldlM(function_t<T1(T3, T2)> const& f, value_type_t<T1> const& z0, T0 const& xs)
{
    using F = T0; using M = T1;
    static_assert(is_monad_v<M>, "Should be a Monad");
    using A = value_type_t<F>;
    using B = value_type_t<M>;
    static_assert(is_same_as_v<T2, A>, "Should be the same");
    static_assert(is_same_as_v<T3, B>, "Should be the same");
    auto const c = _([f](A const& x, function_t<M(B const&)> const& k) {
        return _([f, x, k](B const& z) {
            return f(z, x) >>= k;
        });
    });
    return Foldable_t<F>::foldr(c, _(Monad_t<M>::template mreturn<B>), xs)(z0);
}

//-- | Map each element of a structure to an action, evaluate these
//-- actions from left to right, and ignore the results. For a version
//-- that doesn't ignore the results see 'Data.Traversable.traverse'.
//traverse_ :: (Foldable t, Applicative f) => (a -> f b) -> t a -> f ()
//traverse_ f = foldr c (pure ())
//  -- See Note [List fusion and continuations in 'c']
//  where c x k = f x *> k
FUNCTION_TEMPLATE(3) constexpr FOLDABLE_TYPE(T0, TYPEOF_T(T1, None)) traverse_(function_t<T1(T2)> const& f, T0 const& xs)
{
    using F = T0; using AP = T1;
    static_assert(is_applicative_v<AP>, "Should be an Applicative");
    using A = value_type_t<F>;
    static_assert(is_same_as_v<T2, A>, "Should be the same");
    auto const c = _([f](A const& x, f0<typeof_t<AP, None> > const& k) {
        return _([f, x, k]() {
            AP const fx = f(x);
            return fx *= *k;
        });
    });
    return *Foldable_t<F>::foldr(c, _([]() { return Applicative_t<AP>::pure(None()); }), xs);
}

//-- | 'for_' is 'traverse_' with its arguments flipped. For a version
//-- that doesn't ignore the results see 'Data.Traversable.for'.
//--
//-- >>> for_ [1..4] print
//-- 1
//-- 2
//-- 3
//-- 4
//for_ :: (Foldable t, Applicative f) => t a -> (a -> f b) -> f ()
//for_ = flip traverse_
FUNCTION_TEMPLATE(3) constexpr FOLDABLE_TYPE(T0, TYPEOF_T(T1, None)) for_(T0 const& xs, function_t<T1(T2)> const& f) {
    return traverse_(f, xs);
}

//-- | Map each element of a structure to a monadic action, evaluate
//-- these actions from left to right, and ignore the results. For a
//-- version that doesn't ignore the results see
//-- 'Data.Traversable.mapM'.
//--
//-- As of base 4.8.0.0, 'mapM_' is just 'traverse_', specialized to
//-- 'Monad'.
//mapM_ :: (Foldable t, Monad m) => (a -> m b) -> t a -> m ()
//mapM_ f = foldr c (return ())
//  -- See Note [List fusion and continuations in 'c']
//  where c x k = f x >> k
FUNCTION_TEMPLATE(3) constexpr FOLDABLE_TYPE(T0, TYPEOF_T(T1, None)) mapM_(function_t<T1(T2)> const& f, T0 const& xs)
{
    using F = T0; using M = T1;
    static_assert(is_monad_v<M>, "Should be a Monad");
    using A = value_type_t<F>;
    static_assert(is_same_as_v<T2, A>, "Should be the same");
    auto const c = _([f](A const& x, f0<typeof_t<M, None> > const& k) {
        return _([f, x, k]() {
            M const fx = f(x);
            return fx >> *k;
        });
    });
    return *Foldable_t<F>::foldr(c, _([]() { return Monad_t<M>::mreturn(None()); }), xs);
}

//-- | 'forM_' is 'mapM_' with its arguments flipped. For a version that
//-- doesn't ignore the results see 'Data.Traversable.forM'.
//--
//-- As of base 4.8.0.0, 'forM_' is just 'for_', specialized to 'Monad'.
//forM_ :: (Foldable t, Monad m) => t a -> (a -> m b) -> m ()
//forM_ = flip mapM_
FUNCTION_TEMPLATE(3) constexpr FOLDABLE_TYPE(T0, TYPEOF_T(T1, None)) forM_(T0 const& xs, function_t<T1(T2)> const& f) {
    return mapM_(f, xs);
}

//-- | Evaluate each action in the structure from left to right, and
//-- ignore the results. For a version that doesn't ignore the results
//-- see 'Data.Traversable.sequenceA'.
//sequenceA_ :: (Foldable t, Applicative f) => t (f a) -> f ()
//sequenceA_ = foldr c (pure ())
//  -- See Note [List fusion and continuations in 'c']
//  where c m k = m *> k
template<typename F>
foldable_type<F, typeof_t<value_type_t<F>, None> > sequenceA_(F const& xs)
{
    using AP = value_type_t<F>;
    static_assert(is_applicative_v<AP>, "Should be an Applicative");
    using A = value_type_t<AP>;
    auto const c = _([](AP const& x, f0<typeof_t<AP, None> > const& k){
        return _([x, k](){ return x *= *k; });
    });
    return *Foldable_t<F>::foldr(c, _([](){ return Applicative_t<AP>::pure(None()); }), xs);
}

//-- | Evaluate each monadic action in the structure from left to right,
//-- and ignore the results. For a version that doesn't ignore the
//-- results see 'Data.Traversable.sequence'.
//--
//-- As of base 4.8.0.0, 'sequence_' is just 'sequenceA_', specialized
//-- to 'Monad'.
//sequence_ :: (Foldable t, Monad m) => t (m a) -> m ()
//sequence_ = foldr c (return ())
//  -- See Note [List fusion and continuations in 'c']
//  where c m k = m >> k
template<typename F>
foldable_type<F, typeof_t<value_type_t<F>, None> > sequence_(F const& xs)
{
    using M = value_type_t<F>;
    static_assert(is_monad_v<M>, "Should be a Monad");
    using A = value_type_t<M>;
    auto const c = _([](M const& x, f0<typeof_t<M, None> > const& k){
        return _([x, k](){ return x >> *k; });
    });
    return *Foldable_t<F>::foldr(c, _([](){ return Monad_t<M>::mreturn(None()); }), xs);
}

//-- | The sum of a collection of actions, generalizing 'concat'.
//--
//-- >>> asum [Just "Hello", Nothing, Just "World"]
//-- Just "Hello"
//asum :: (Foldable t, Alternative f) => t (f a) -> f a
//asum = foldr (<|>) empty
template<typename F>
foldable_type<F, value_type_t<F> > asum(F const& f)
{
    using ALT = value_type_t<F>;
    static_assert(is_alternative_v<ALT>, "Should be an Alternative");
    using A = value_type_t<ALT>;
    return Foldable_t<F>::foldr(_(Alternative_t<ALT>::template alt_op<A>), Alternative_t<ALT>::template empty<A>(), f);
}

//-- | The sum of a collection of actions, generalizing 'concat'.
//-- As of base 4.8.0.0, 'msum' is just 'asum', specialized to 'MonadPlus'.
//msum :: (Foldable t, MonadPlus m) => t (m a) -> m a
//msum = asum
template<typename F>
foldable_type<F, value_type_t<F> > msum(F const& f){
    static_assert(is_monad_plus_v<value_type_t<F> >, "Should be a MonadPlus");
    return asum(f);
}

//-- | The concatenation of all the elements of a container of lists.
//concat :: Foldable t => t [a] -> [a]
//concat xs = build (\c n -> foldr (\x y -> foldr c y x) n xs)
template<typename F>
foldable_type<F, value_type_t<F> > concat(F const& xs)
{
    static_assert(is_list_v<value_type_t<F> >, "Should be a List");
    using A = value_type_t<value_type_t<F> >;
    return build(_([&xs](function_t<List<A>(A const&, List<A> const&)> const& c, List<A> const& n){
        return Foldable_t<F>::foldr(_([&xs, c, n](List<A> const& x, List<A> const& y){
            return Foldable<_List>::foldr(c, y, x); }), n, xs); }));
}

//-- | Map a function over all the elements of a container and concatenate
//-- the resulting lists.
//concatMap :: Foldable t => (a -> [b]) -> t a -> [b]
//concatMap f xs = build (\c n -> foldr (\x b -> foldr c b (f x)) n xs)
FUNCTION_TEMPLATE(3) constexpr FOLDABLE_TYPE(T0, List<T1>) concatMap(function_t<List<T1>(T2)> const& f, T0 const& xs)
{
    using F = T0; using B = T1;
    static_assert(is_same_as_v<T2, value_type_t<F> >, "Should be the same");
    return build(_([&f, &xs](function_t<List<B>(B const&, List<B> const&)> const& c, List<B> const& n) {
        return Foldable_t<F>::foldr(_([&f, &xs, c, n](value_type_t<F> const& x, List<B> const& b) {
            return Foldable<_List>::foldr(c, b, f(x)); }), n, xs); }));
}

_FUNCPROG_END
