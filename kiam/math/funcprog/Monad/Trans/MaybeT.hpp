#pragma once

#include "../../Maybe.hpp"
#include "../../Alternative.hpp"
#include "MonadTrans.hpp"
#include "../MonadReader.hpp"
#include "../MonadWriter.hpp"
#include "../MonadState.hpp"
#include "../Signatures.hpp"

_FUNCPROG_BEGIN

/*
-- | The parameterizable maybe monad, obtained by composing an arbitrary
-- monad with the 'Maybe' monad.
--
-- Computations are actions that may produce a value or exit.
--
-- The 'return' function yields a computation that produces that
-- value, while @>>=@ sequences two subcomputations, exiting if either
-- computation does.
newtype MaybeT m a = MaybeT { runMaybeT :: m (Maybe a) }
*/

template<class _M>
struct _MaybeT;

template<class _M, typename A>
struct MaybeT;

#define MAYBET_(_M, A) BOOST_IDENTITY_TYPE((MaybeT<_M, A>))
#define MAYBET(_M, A) typename MAYBET_(_M, A)

struct __MaybeT
{
    template<class _M>
    using mt_type = _MaybeT<_M>;

    // -- | Transform the computation inside a @MaybeT@.
    // mapMaybeT :: (m (Maybe a) -> n (Maybe b)) -> MaybeT m a -> MaybeT n b
    // mapMaybeT f = MaybeT . f . runMaybeT
    template<typename MA, typename NB>
    static MaybeT<base_class_t<NB>, value_type_t<value_type_t<NB> > >
    map(function_t<NB(MA const&)> const& f, MaybeT<base_class_t<MA>, value_type_t<value_type_t<MA> > > const& m) {
        return typeof_t<NB, remove_f0_t<value_type_t<NB> > >(f(m.run()));
    }

};

template<typename MA, typename NB>
MaybeT<base_class_t<NB>, value_type_t<value_type_t<NB> > >
mapMaybeT(function_t<NB(MA const&)> const& f, MaybeT<base_class_t<MA>, value_type_t<value_type_t<MA> > > const& m) {
    return __MaybeT::map(f, m);
}

template<typename MA, typename NB>
function_t<MaybeT<base_class_t<NB>, value_type_t<value_type_t<NB> > >(
    MaybeT<base_class_t<MA>, value_type_t<value_type_t<MA> > > const&
)> _mapMaybeT(function_t<NB(MA const&)> const& f) {
    return [f](MaybeT<base_class_t<MA>, value_type_t<value_type_t<MA> > > const& m) {
        return mapMaybeT(f, m);
    };
}

template<class _M>
struct _MaybeT
{
    static_assert(is_monad<_M>::value, "Should be a monad");

    using base_class = _MaybeT;

    template<typename A>
    using type = MaybeT<_M, A>;

    /*
    -- | Lift a @listen@ operation to the new monad.
    liftListen :: (Monad m) => Listen w m (Maybe a) -> Listen w (MaybeT m) a
    liftListen listen = mapMaybeT $ \ m -> do
        (a, w) <- listen m
        return $! fmap (\ r -> (r, w)) a
    */
    template<typename W, typename A>
    static Listen<W, _MaybeT<_M>, A> liftListen(Listen<W, _M, Maybe<A> > const& listen) {
        return _mapMaybeT(_([listen](typename _M::template type<Maybe<A> > const& m) {
            return _do(p, listen(m),
                return Monad<_M>::mreturn(fmap(_([&p](A const& r) { return pair_t<A, W>(r, snd(p)); }), fst(p))););
        }));
    }
    /*
    -- | Lift a @pass@ operation to the new monad.
    liftPass :: (Monad m) => Pass w m (Maybe a) -> Pass w (MaybeT m) a
    liftPass pass = mapMaybeT $ \ m -> pass $ do
        a <- m
        return $! case a of
            Nothing     -> (Nothing, id)
            Just (v, f) -> (Just v, f)
    */
    template<typename W, typename A>
    static Pass<W, _MaybeT<_M>, A> liftPass(Pass<W, _M, Maybe<A> > const& pass)
    {
        using WF = function_t<W(W const&)>;
        using pair_type = pair_t<Maybe<A>, WF>;
        return _mapMaybeT(_([pass](typename _M::template type<Maybe<pair_t<A, WF> > > const& m) {
            return pass(_do(p, m,
                return Monad<_M>::mreturn(p ? pair_type(Just(fst(p.value())), snd(p.value())) : pair_type(Nothing<A>(), _(id<W>)));));
        }));
    }
};

template<class _M, typename A>
struct MaybeT : _MaybeT<_M>
{
    using super = _MaybeT<_M>;
    using value_type = A;
    using value_t = typename _M::template type<Maybe<A> >;

    MaybeT(value_t const& value) : value(value) {}
    MaybeT(value_type const& value) : value(Just(value)) {}

    value_t const& run() const {
        return value;
    }

private:
    const value_t value;
};

template<typename _M, typename A>
MaybeT<_M, A> MaybeT_(typename _M::template type<Maybe<A> > const& value) {
    return value;
}

template<class _M, typename A>
typename _M::template type<Maybe<A> > runMaybeT(MaybeT<_M, A> const& x) {
    return x.run();
}

template<typename T>
struct is_MaybeT : std::false_type {};

template<class _M, typename A>
struct is_MaybeT<MaybeT<_M, A> > : std::true_type {};

// Functor
template<class _M>
struct is_functor<_MaybeT<_M> > : is_functor<_M> {};

template<typename _M>
struct is_same_functor<_MaybeT<_M>, _MaybeT<_M> > : is_functor<_M> {};

template<class _M>
struct Functor<_MaybeT<_M> >
{
    template<typename A>
    using Maybe_type = Maybe<fdecay<A> >;

    // <$> fmap :: Functor f => (a -> b) -> f a -> f b
    // fmap :: (a -> b) -> MaybeT a -> MaybeT b
    // fmap f = mapMaybeT (fmap (fmap f))
    template<typename Ret, typename Arg, typename... Args>
    static MaybeT<_M, remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, MaybeT<_M, fdecay<Arg> > const& v){
        return mapMaybeT(_fmap<typename _M::template type<Maybe_type<Arg> > >(_fmap<Maybe_type<Arg> >(f)), v);
    }
};

// Applicative
template<class _M>
struct is_applicative<_MaybeT<_M> > : is_monad<_M> {};

template<class _M>
struct is_same_applicative<_MaybeT<_M>, _MaybeT<_M> > : is_applicative<_M> {};

template<class _M>
struct Applicative<_MaybeT<_M> > : Functor<_MaybeT<_M> >
{
    using base_class = _MaybeT<_M>;
    using super = Functor<base_class>;

    // pure = MaybeT . return . Just
    template<typename A>
    static MaybeT<_M, A> pure(A const& x) {
        return Monad<_M>::mreturn(Just(x));
    }

    /*
        mf <*> mx = MaybeT $ do
        mb_f <- runMaybeT mf
        case mb_f of
            Nothing -> return Nothing
            Just f  -> do
                mb_x <- runMaybeT mx
                case mb_x of
                    Nothing -> return Nothing
                    Just x  -> return (Just (f x))
    */
    template<typename Ret, typename Arg, typename... Args>
    static MaybeT<_M, remove_f0_t<function_t<Ret(Args...)> > >
    apply(MaybeT<_M, function_t<Ret(Arg, Args...)> > const& mf, MaybeT<_M, fdecay<Arg> > const& mx)
    {
        using return_type = remove_f0_t<function_t<Ret(Args...)> >;
        return _do(mb_f, mf.run(),
            return mb_f ? _do(mb_x, mx.run(),
                return Monad<_M>::mreturn(mb_x ? Just(mb_f.value()(mb_x.value())) : Nothing<return_type>());
            ) : Monad<_M>::mreturn(Nothing<return_type>()););
    }
};

// m *> k = m >>= \_ -> k
template<class _M, typename Fa, typename Fb>
MaybeT<_M, Fb> operator*=(MaybeT<_M, Fa> const& m, MaybeT<_M, Fb> const& k) {
    return m >>= _([&k](Fa const&) { return k; });
}

/*
instance (Monad m) => Monad (MaybeT m) where
#if !(MIN_VERSION_base(4,8,0))
    return = MaybeT . return . Just
    {-# INLINE return #-}
#endif
    x >>= f = MaybeT $ do
        v <- runMaybeT x
        case v of
            Nothing -> return Nothing
            Just y  -> runMaybeT (f y)
    {-# INLINE (>>=) #-}
*/
// Monad
template<class _M>
struct is_monad<_MaybeT<_M> > : is_monad<_M> {};

template<class _M>
struct is_same_monad<_MaybeT<_M>, _MaybeT<_M> > : is_monad<_M> {};

template<class _M>
struct Monad<_MaybeT<_M> > : Applicative<_MaybeT<_M> >
{
    using base_class = _MaybeT<_M>;
    using super = Applicative<base_class>;

    template<typename T>
    using liftM_type = MaybeT<_M, T>;

    // return = MaybeT . return
    template<typename A>
    static MaybeT<_M, A> mreturn(A const& x) {
        return Monad<_M>::mreturn(Just(x));
    }

    // fail _ = MaybeT (return Nothing)
    static MaybeT<_M, const char*> fail(const char*) {
        return Monad<_M>::mreturn(Nothing<const char*>());
    }

    /*
    x >>= f = MaybeT $ do
        v <- runMaybeT x
        case v of
            Nothing -> return Nothing
            Just y  -> runMaybeT (f y)
    */
    template<typename Ret, typename Arg, typename... Args>
    static remove_f0_t<function_t<MaybeT<_M, Ret>(Args...)> >
    mbind(MaybeT<_M, fdecay<Arg>> const& x, function_t<MaybeT<_M, Ret>(Arg, Args...)> const& f){
        return invoke_f0(_([x, f](Args... args) {
            return _do(v, x.run(),
                return v ? f(v.value(), args...).run() : Monad<_M>::mreturn(Nothing<value_type_t<MaybeT<_M, Ret> > >()););
        }));
    }
};

/*
instance (Monad m) => MonadPlus (MaybeT m) where
    mzero = MaybeT (return Nothing)
    {-# INLINE mzero #-}
    mplus x y = MaybeT $ do
        v <- runMaybeT x
        case v of
            Nothing -> runMaybeT y
            Just _  -> return v
    {-# INLINE mplus #-}
*/
template<class _M>
struct is_monad_plus<_MaybeT<_M> > : is_monad<_M> {};

template<class _M>
struct is_same_monad_plus<_MaybeT<_M>, _MaybeT<_M> > : is_monad_plus<_M> {};

template<class _M>
struct MonadPlus<_MaybeT<_M> > : Monad<_MaybeT<_M> >
{
    using base_class = _MaybeT<_M>;
    using super = Monad<base_class>;

    template<typename A>
    static MaybeT<_M, A> mzero() {
        return Monad<_M>::mreturn(Nothing<A>());
    }

    template<typename T>
    struct mplus_result_type;

    template<typename T>
    using mplus_result_type_t = typename mplus_result_type<T>::type;

    template<typename A>
    struct mplus_result_type<MaybeT<_M, A> >
    {
        using type = MaybeT<_M, A>;
    };

    template<typename A>
    static MaybeT<_M, A> mplus(MaybeT<_M, A> const& x, MaybeT<_M, A> const& y){
        return _do(v, x.run(), return v ? v : y.run(););
    }
};

template<class _M, typename A>
MaybeT<_M, A> mplus(MaybeT<_M, A> const& x, MaybeT<_M, A> const& y) {
    return MonadPlus<_MaybeT<_M> >::mplus(x, y);
}

/*
instance (Functor m, Monad m) => Alternative (MaybeT m) where
    empty = MaybeT (return Nothing)
    {-# INLINE empty #-}
    x <|> y = MaybeT $ do
        v <- runMaybeT x
        case v of
            Nothing -> runMaybeT y
            Just _  -> return v
    {-# INLINE (<|>) #-}
*/
template<class _M>
struct is_alternative<_MaybeT<_M> > : is_monad<_M> {};

template<class _M>
struct is_same_alternative<_MaybeT<_M>, _MaybeT<_M> > : is_alternative<_M> {};

template<class _M>
struct Alternative<_MaybeT<_M> >
{
    template<typename A>
    static MaybeT<_M, A> empty() {
        return Monad<_M>::mreturn(Nothing<A>());
    }

    template<typename T>
    struct alt_op_result_type;

    template<typename T>
    using alt_op_result_type_t = typename alt_op_result_type<T>::type;

    template<typename A>
    struct alt_op_result_type<MaybeT<_M, A> >
    {
        using type = MaybeT<_M, A>;
    };

    template<typename A>
    static MaybeT<_M, A> alt_op(MaybeT<_M, A> const& x, MaybeT<_M, A> const& y){
        return _do(v, x.run(), return v ? v : y.run(););
    }
};

template<class _M, typename A>
MaybeT<_M, A> operator|(MaybeT<_M, A> const& x, MaybeT<_M, A> const& y) {
    return Alternative<_MaybeT<_M> >::alt_op(x, y);
}

// Foldable
template<class _M, typename A>
struct is_foldable<MaybeT<_M, A> > : is_foldable<typename _M::template type<Maybe<A> > > {};

template<class _M>
struct Foldable<_MaybeT<_M> >
{
    // foldMap :: Monoid m => (a -> m) -> t a -> m
    // foldMap f (MaybeT a) = foldMap (foldMap f) a
    template<typename M, typename Arg>
    static typename std::enable_if<is_monoid_t<M>::value, M>::type
    foldMap(function_t<M(Arg)> const& f, MaybeT<_M, fdecay<Arg> > const& x){
        return Foldable<_M>::foldMap(_foldMap<Maybe<fdecay<Arg> > >(f), x.run());
    }
};

// Traversable
template<class _M, typename A>
struct is_traversable<MaybeT<_M, A> > : std::true_type {};

template<class _M, typename A1, typename A2>
struct is_same_traversable<MaybeT<_M, A1>, MaybeT<_M, A2> > : is_same_traversable<typename _M::template type<Maybe<A1> >, typename _M::template type<Maybe<A2> > > {};

template<class _M>
struct Traversable<_MaybeT<_M> >
{
    // traverse f (MaybeT a) = MaybeT <$> traverse (traverse f) a
    template<typename AP, typename Arg>
    static typename std::enable_if<is_applicative_t<AP>::value, typeof_t<AP, MaybeT<_M, fdecay<Arg> > > >::type
    traverse(function_t<AP(Arg)> const& f, MaybeT<_M, fdecay<Arg> > const& x){
        return _(MaybeT_<_M, fdecay<Arg> >) / Traversable<_M>::traverse(_traverse<Maybe<fdecay<Arg> > >(f), x.run());
    }
};

// MonadZip
template<class _M>
struct MonadZip<_MaybeT<_M> > : _MonadZip<MonadZip<_MaybeT<_M> > >
{
    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith f (MaybeT a) (MaybeT b) = MaybeT $ mzipWith (liftA2 f) a b
    template<typename C, typename A, typename B>
    static MaybeT<_M, C> mzipWith(function_t<C(A, B)> const& f,
        MaybeT<_M, fdecay<A> > const& ma, MaybeT<_M, fdecay<B> > const& mb)
    {
        return MonadZip<_M>::mzipWith(_liftA2<Maybe<fdecay<B> >, Maybe<fdecay<A> > >(f), ma.run(), mb.run());
    }
};

// MonadTrans
template<class _M>
struct MonadTrans<_MaybeT<_M> >
{
    // lift = MaybeT . liftM Just
    template<typename F>
    static MaybeT<_M, value_type_t<F> > lift(F const& x){
        return liftM(_(Just<value_type_t<F> >), x);
    }
};

// MonadReader
template<typename R, class _M>
struct MonadReader<R, _MaybeT<_M> > : _MonadReader<R, _MaybeT<_M>, MonadReader<R, _MaybeT<_M> > >
{
    using base_class = _MaybeT<_M>;
    using super = _MonadReader<R, base_class, MonadReader<R, base_class> >;

    template<typename T>
    using type = typename super::template type<T>;

    // ask = lift ask
    static type<R> ask() {
        return MonadTrans<base_class>::lift(MonadReader<R, _M>::ask());
    }

    // local = mapMaybeT . local
    template<typename A, typename RArg>
    static typename std::enable_if<is_same_as<R, RArg>::value, type<A> >::type
    local(function_t<R(RArg)> const& f, type<A> const& m)
    {
        using FA = typename _M::template type<Maybe<A> >;
        return (_(mapMaybeT<FA, FA>) & _(MonadReader<R, _M>::template local<Maybe<A>, RArg>))(f, m);
    }

    // reader = lift . reader
    template<typename A, typename RArg>
    static typename std::enable_if<is_same_as<R, RArg>::value, type<A> >::type
    reader(function_t<A(RArg)> const& f){
        return MonadTrans<base_class>::lift(MonadReader<R, _M>::reader(f));
    }
};

// MonadWriter
template<typename W, class _M>
struct MonadWriter<W, _MaybeT<_M> > : _MonadWriter<W, _MaybeT<_M>, MonadWriter<W, _MaybeT<_M> > >
{
    using base_class = _MaybeT<_M>;
    using super = _MonadWriter<W, base_class, MonadWriter<W, base_class> >;

    template<typename T>
    using type = typename super::template type<T>;

    // writer = lift . writer
    template<typename A>
    static type<A> writer(pair_t<A, W> const& p) {
        return MonadTrans<base_class>::lift(MonadWriter<W, _M>::writer(p));
    }

    // tell = lift . tell
    static type<None> tell(W const& w) {
        return MonadTrans<base_class>::lift(MonadWriter<W, _M>::tell(w));
    }

    // listen = Maybe.liftListen listen
    template<typename A>
    static type<pair_t<A, W> > listen(MaybeT<_M, A> const& m) {
        return _MaybeT<_M>::template liftListen<W, A>(_(MonadWriter<W, _M>::template listen<Maybe<A> >))(m);
    }

    // pass = Maybe.liftPass pass
    template<typename A>
    static type<A> pass(MaybeT<_M, pair_t<A, function_t<W(W const&)> > > const& m) {
        return _MaybeT<_M>::template liftPass<W, A>(_(MonadWriter<W, _M>::template pass<Maybe<A> >))(m);
    }
};

// MonadState
template<typename S, typename _M>
struct MonadState<S, _MaybeT<_M> > : _MonadState<S, _MaybeT<_M>, MonadState<S, _MaybeT<_M> > >
{
    using base_class = _MaybeT<_M>;
    using super = _MonadState<S, base_class, MonadState>;

    template<typename T>
    using type = typename super::template type<T>;

    // state = lift . state
    template<typename A>
    static type<A> state(function_t<pair_t<A, S>(S const&)> const& f) {
        return MonadTrans<base_class>::lift(MonadState<S, _M>::state(f));
    }

    // get = lift get
    static type<S> get() {
        return MonadTrans<base_class>::lift(MonadState<S, _M>::get());
    }

    // put = lift . put
    static type<None> put(S const& s) {
        return MonadTrans<base_class>::lift(MonadState<S, _M>::put(s));
    }
};

_FUNCPROG_END

namespace std {

    template<class _M, typename A>
    ostream& operator<<(ostream& os, _FUNCPROG::MaybeT<_M, A> const& v) {
        return os << "MaybeT[" << v.run() << ']';
    }

}
