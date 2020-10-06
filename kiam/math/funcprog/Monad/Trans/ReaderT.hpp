#pragma once

#include "../../Identity.hpp"
#include "../MonadReader.hpp"
#include "../MonadWriter.hpp"
#include "../MonadState.hpp"

_FUNCPROG_BEGIN

/*
-- | The parameterizable reader monad.
--
-- Computations are functions of a shared environment.
--
-- The 'return' function ignores the environment, while @>>=@ passes
-- the inherited environment to both subcomputations.
*/
template<typename R, typename _M>
struct _ReaderT;

template<typename R>
using _Reader = _ReaderT<R, _Identity>;

template<typename R, typename _M, typename A>
struct ReaderT;

#define READERT_(R, _M, A) BOOST_IDENTITY_TYPE((ReaderT<R, _M, A>))
#define READERT(R, _M, A) typename READERT_(R, _M, A)

template<typename R, typename A>
using Reader = ReaderT<R, _Identity, A>;

#define READER_(R, A) BOOST_IDENTITY_TYPE((Reader<R, A>))
#define READER(R, A) typename READER_(R, A)

/*
-- | Constructor for computations in the reader monad (equivalent to 'asks').
reader :: (Monad m) => (r -> a) -> ReaderT r m a
reader f = ReaderT (return . f)
*/
template<typename _M, typename Ret, typename RArg>
ReaderT<fdecay<RArg>, _M, Ret> reader(function_t<Ret(RArg)> const& f) {
    return _ReaderT<fdecay<RArg>, _M>::reader(f);
}

template<typename R>
struct __ReaderT
{
    template<typename _M>
    using mt_type = _ReaderT<R, _M>;
};

template<typename R, typename _M>
struct _ReaderT
{
    static_assert(_is_monad<_M>::value, "Should be a monad");
    using base_class = _ReaderT;

    template<typename A>
    using type = ReaderT<R, _M, A>;

    // -- | Fetch the value of the environment.
    // ask :: (Monad m) => ReaderT r m r
    // ask = ReaderT return
    static ReaderT<R, _M, R> ask() {
        return _(Monad<_M>::template mreturn<R>);
    }

    // -- | Execute a computation in a modified environment
    // -- (a specialization of 'withReaderT').
    // --
    // -- * @'runReaderT' ('local' f m) = 'runReaderT' m . f@
    // local
    //   :: (r -> r)         -- ^ The function to modify the environment.
    //   -> ReaderT r m a    -- ^ Computation to run in the modified environment.
    //   -> ReaderT r m a
    // local = withReaderT
    template<typename A, typename RArg>
    static typename std::enable_if<is_same_as<R, RArg>::value, ReaderT<R, _M, A> >::type
    local(function_t<R(RArg)> const& f, ReaderT<R, _M, A> const& m){
        return withReaderT(f, m);
    }

    // | Retrieve a function of the current environment.
    //
    // * @'asks' f = 'liftM' f 'ask'@
    // asks :: (Monad m)
    //   => (r -> a)         -- ^ The selector function to apply to the environment.
    //   -> ReaderT r m a
    // asks f = ReaderT (return . f)
    template<typename A, typename RArg>
    static typename std::enable_if<is_same_as<R, RArg>::value, ReaderT<R, _M, A> >::type
    asks(function_t<A(RArg)> const& f){
        return _(Monad<_M>::template mreturn<A>) & f;
    }

    // | Constructor for computations in the reader monad(equivalent to 'asks').
    //  reader :: (Monad m) = > (r -> a) -> ReaderT r m a
    //  reader f = ReaderT (return . f)
    template<typename A, typename RArg>
    static typename std::enable_if<is_same_as<R, RArg>::value, ReaderT<R, _M, A> >::type
    reader(function_t<A(RArg)> const& f) {
        return asks(f);
    }
};

template<typename R, typename _M, typename A>
struct ReaderT : _ReaderT<R, _M>
{
    using value_type = A;
    using return_type = typename _M::template type<value_type>;
    using function_type = function_t<return_type(R const&)>;

    ReaderT(function_type const& func) : func(func) {}
    ReaderT(function_t<return_type(R)> const& f) : func([f](R const& value) { return f(value); }) {}

    return_type run(R const& r) const {
        return func(r);
    }

private:
    const function_type func;
};

template<typename R, class _M, typename A>
typename std::enable_if<_is_monad<_M>::value, typename _M::template type<A> >::type
runReaderT(ReaderT<R, _M, A> const& m, R const& r) {
    return m.run(r);
}

template<typename R, class _M, typename A>
typename std::enable_if<_is_monad<_M>::value, function_t<typename _M::template type<A>(R const&)> >::type
_runReaderT(ReaderT<R, _M, A> const& m) {
    return [m](R const& r) { return runReaderT(m, r); };
}

//-- | Transform the computation inside a @ReaderT@.
//-- * @'runReaderT' ('mapReaderT' f m) = f . 'runReaderT' m@
//mapReaderT :: (m a -> n b) -> ReaderT r m a -> ReaderT r n b
//mapReaderT f m = ReaderT $ f . runReaderT m
template<typename R, typename MA, typename NB>
ReaderT<R, base_class_t<NB>, value_type_t<NB> >
mapReaderT(function_t<NB(MA const&)> const& f, ReaderT<R, base_class_t<MA>, value_type_t<MA> > const& m) {
    return f & _runReaderT(m);
}

template<typename R, typename MA, typename NB>
function_t<ReaderT<R, base_class_t<NB>, value_type_t<NB> >(ReaderT<R, base_class_t<MA>, value_type_t<MA> > const&)>
_mapReaderT(function_t<NB(MA const&)> const& f) {
    return [f](ReaderT<R, base_class_t<MA>, value_type_t<MA> > const& m) {
        return mapReaderT(f, m);
    };
}

/*
-- | Execute a computation in a modified environment
-- (a more general version of 'local').
--
-- * @'runReaderT' ('withReaderT' f m) = 'runReaderT' m . f@
withReaderT
    :: (r' -> r)        -- ^ The function to modify the environment.
    -> ReaderT r m a    -- ^ Computation to run in the modified environment.
    -> ReaderT r' m a
withReaderT f m = ReaderT $ runReaderT m . f
*/
template<typename R, typename _M, typename RArg, typename A>
static typename std::enable_if<is_same_as<R, RArg>::value, ReaderT<R, _M, A> >::type
withReaderT(function_t<R(RArg)> const& f, ReaderT<R, _M, A> const& m) {
    return _runReaderT(m) & f;
}

template<typename R, typename _M, typename RArg, typename A>
static typename std::enable_if<is_same_as<R, RArg>::value, function_t<ReaderT<R, _M, A>(ReaderT<R, _M, A> const&)> >::type
_withReaderT(function_t<R(RArg)> const& f) {
    return [f](ReaderT<fdecay<RArg>, _M, A> const& m) {
        return withReaderT(f, m);
    };
}

// liftReaderT :: m a -> ReaderT r m a
// liftReaderT m = ReaderT (const m)
template<typename R, typename MA>
ReaderT<R, base_class_t<MA>, value_type_t<MA> > liftReaderT(MA const& m) {
    return _const_<R>(m);
}

/*
-- | Runs a @Reader@ and extracts the final value from it.
-- (The inverse of 'reader'.)
runReader
    :: Reader r a       -- ^ A @Reader@ to run.
    -> r                -- ^ An initial environment.
    -> a
runReader m = runIdentity . runReaderT m
*/
template<typename R, typename A>
A runReader(Reader<R, A> const& m, R const& r) {
    return (_(runIdentity<A>) & _runReaderT(m))(r);
}

template<typename R, typename A>
function_t<A(R const&)> _runReader(Reader<R, A> const& m) {
    return [m](R const& r) { return runReader(m, r); };
}

/*
-- | Transform the value returned by a @Reader@.
--
-- * @'runReader' ('mapReader' f m) = f . 'runReader' m@
mapReader :: (a -> b) -> Reader r a -> Reader r b
mapReader f = mapReaderT (Identity . f . runIdentity)
*/
template<typename R, typename Arg, typename B>
Reader<R, B> mapReader(function_t<B(Arg)> const& f, Reader<R, fdecay<Arg> > const& m) {
    return mapReaderT(_(Identity_<B>) & f & _(runIdentity<fdecay<Arg> >), m);
}

template<typename R, typename Arg, typename B>
function_t<Reader<R, B>(Reader<R, fdecay<Arg> > const&)>
_mapReader(function_t<B(Arg)> const& f) {
    return [f](Reader<R, fdecay<Arg> > const& m) {
        return mapReader(f, m);
    };
}

/*
-- | Execute a computation in a modified environment
-- (a specialization of 'withReaderT').
--
-- * @'runReader' ('withReader' f m) = 'runReader' m . f@
withReader
    :: (r' -> r)        -- ^ The function to modify the environment.
    -> Reader r a       -- ^ Computation to run in the modified environment.
    -> Reader r' a
withReader = withReaderT
*/
template<typename R1, typename A, typename RArg>
Reader<R1, A> withReader(function_t<R1(RArg)> const& f, Reader<fdecay<RArg>, A> const& m) {
    return withReaderT(f, m);
}

template<typename R1, typename A, typename RArg>
function_t<Reader<R1, A>(Reader<fdecay<RArg>, A> const&)>
_withReader(function_t<R1(RArg)> const& f) {
    return [f](Reader<fdecay<RArg>, A> const& m) {
        return withReader(f, m);
    };
}

template<typename T>
struct is_reader : std::false_type {};

template<typename R, typename _M, typename A>
struct is_reader<ReaderT<R, _M, A> > : std::true_type {};

// Functor
template<typename R, typename _M>
struct _is_functor<_ReaderT<R, _M> > : _is_functor<_M> {};

template<typename R, typename _M>
struct Functor<_ReaderT<R, _M> >
{
    // <$> fmap :: Functor f => (a -> b) -> f a -> f b
    // fmap f  = mapReaderT (fmap f)
    template<typename Ret, typename Arg, typename... Args>
    static ReaderT<R, _M, remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, ReaderT<R, _M, fdecay<Arg> > const& v){
        return mapReaderT(_(Functor<_M>::template fmap<Ret, Arg, Args...>) << f, v);
    }
};

// (<$) ::a->f b->f a
// x <$ v = mapReaderT (x <$) v
template<typename R, typename A, typename _M, typename B>
ReaderT<R, _M, A> left_fmap(A const& x, ReaderT<R, _M, B> const& v) {
    return mapReaderT(_left_fmap<typename _M::template type<B> >(x), v);
}

template<typename R, typename A, typename _M, typename B>
ReaderT<R, _M, A> operator/=(A const& x, ReaderT<R, _M, B> const& v) {
    return left_fmap<R, A, _M, B>(x, v);
}

// Applicative
template<typename R, typename _M>
struct _is_applicative<_ReaderT<R, _M> > : _is_applicative<_M> {};

template<typename R, typename _M>
struct Applicative<_ReaderT<R, _M> > : Functor<_ReaderT<R, _M> >
{
    using super = Functor<_ReaderT<R, _M> >;

    // pure = liftReaderT . pure
    template<typename A>
    static ReaderT<R, _M, A> pure(A const& x) {
        return liftReaderT<R>(Applicative<_M>::pure(x));
    }

    // f <*> v = ReaderT $ \ r -> runReaderT f r <*> runReaderT v r
    template<typename Ret, typename Arg, typename... Args>
    static ReaderT<R, _M, remove_f0_t<function_t<Ret(Args...)> > >
    apply(ReaderT<R, _M, function_t<Ret(Arg, Args...)> > const& f, ReaderT<R, _M, fdecay<Arg> > const& v){
        return _([f, v](R const& r){
            return Applicative<_M>::apply(f.run(r), v.run(r));
        });
    }
};

// liftA2 f x y = ReaderT $ \r -> liftA2 f (runReaderT x r) (runReaderT y r)
template<typename R, typename _M, typename Ret, typename ArgX, typename ArgY, typename... Args>
ReaderT<R, _M, remove_f0_t<function_t<Ret(Args...)> > >
liftA2(function_t<Ret(ArgX, ArgY, Args...)> const& f, ReaderT<R, _M, fdecay<ArgX> > const& x, ReaderT<R, _M, fdecay<ArgY> > const& y){
    return _([f, x, y](R const& r){
        return liftA2(f, x.run(r), y.run(r));
    });
}

// u *> v = ReaderT $ \ r -> runReaderT u r *> runReaderT v r
template<typename R, typename _M, typename Fa, typename Fb>
ReaderT<R, _M, Fb> operator*=(ReaderT<R, _M, Fa> const& a, ReaderT<R, _M, Fb> const& b) {
    return _([a, b](R const& r) { return a.run(r) *= b.run(r); });
}

// u <* v = ReaderT $ \ r -> runReaderT u r <* runReaderT v r
template<typename R, typename _M, typename Fa, typename Fb>
ReaderT<R, _M, Fa> operator^=(ReaderT<R, _M, Fa> const& a, ReaderT<R, _M, Fb> const& b) {
    return _([a, b](R const& r) { return a.run(r) ^= b.run(r); });
}

// Monad
template<typename R, typename _M>
struct _is_monad<_ReaderT<R, _M> > : _is_monad<_M> {};

template<typename R, typename _M>
struct Monad<_ReaderT<R, _M> > : Applicative<_ReaderT<R, _M> >
{
    using super = Applicative<_ReaderT<R, _M> >;

    template<typename T>
    using liftM_type = ReaderT<R, _M, T>;

    // return = lift . return
    template<typename A>
    static ReaderT<R, _M, A> mreturn(A const& x) {
        return liftReaderT<R>(Monad<_M>::mreturn(x));
    }

    // fail msg = lift (fail msg)
    static ReaderT<R, _M, const char*> fail(const char* msg) {
        return liftReaderT<R>(Monad<_M>::fail(msg));
    }

    // m >>= k  = ReaderT $ \ r -> do
    //   a <- runReaderT m r
    //   runReaderT (k a) r
    template<typename Ret, typename Arg, typename... Args>
    static remove_f0_t<function_t<ReaderT<R, _M, Ret>(Args...)> >
    mbind(ReaderT<R, _M, fdecay<Arg> > const& m, function_t<ReaderT<R, _M, Ret>(Arg, Args...)> const& f){
        return invoke_f0(_([m, f](Args... args) {
            return _([m, f, args...](R const& r){
                return _do(a, m.run(r), return f(a, args...).run(r););
            });
        }));
    }
};

// (>>) = (*>)
template<typename R, typename _M, typename Fa, typename Fb>
ReaderT<R, _M, Fb> operator>>(ReaderT<R, _M, Fa> const& a, ReaderT<R, _M, Fb> const& b) {
    return a *= b;
}

// MonadPlus
template<typename R, typename _M>
struct _is_monad_plus<_ReaderT<R, _M> > : _is_monad_plus<_M> {};

template<typename R, typename _M>
struct MonadPlus<_ReaderT<R, _M> > : Monad<_ReaderT<R, _M> >
{
    using super = Monad<_ReaderT<R, _M> >;

    // mzero = lift mzero
    template<typename A>
    static ReaderT<R, _M, A> mzero() {
        return liftReaderT<R>(MonadPlus<_M>::template mzero<A>());
    }

    template<typename T>
    struct mplus_result_type;

    template<typename T>
    using mplus_result_type_t = typename mplus_result_type<T>::type;

    template<typename A>
    struct mplus_result_type<ReaderT<R, _M, A> >
    {
        using type = ReaderT<R, _M, A>;
    };

    // m `mplus` n = ReaderT $ \ r -> runReaderT m r `mplus` runReaderT n r
    template<typename A>
    static ReaderT<R, _M, A> mplus(ReaderT<R, _M, A> const& m, ReaderT<R, _M, A> const& n) {
        return _([m, n](R const& r) { return MonadPlus<_M>::mplus(m.run(r), n.run(r)); });
    }
};

// Alternative
template<typename R, typename _M>
struct _is_alternative<_ReaderT<R, _M> > : _is_alternative<_M> {};

template<typename R, typename _M>
struct Alternative<_ReaderT<R, _M> >
{
    // empty = liftReaderT empty
    template<typename A>
    static ReaderT<R, _M, A> empty() {
        return liftReaderT<R>(Alternative<_M>::template empty<A>());
    }

    template<typename T>
    struct alt_op_result_type;

    template<typename T>
    using alt_op_result_type_t = typename alt_op_result_type<T>::type;

    template<typename A>
    struct alt_op_result_type<ReaderT<R, _M, A> >
    {
        using type = ReaderT<R, _M, A>;
    };

    // m <|> n = ReaderT $ \ r -> runReaderT m r <|> runReaderT n r
    template<typename A>
    static ReaderT<R, _M, A> alt_op(ReaderT<R, _M, A> const& m, ReaderT<R, _M, A> const& n){
        return _([m, n](R const& r) { return m.run(r) | n.run(r); });
    }
};

// MonadZip
template<typename R, typename _M>
struct MonadZip<_ReaderT<R, _M> > : _MonadZip<MonadZip<_ReaderT<R, _M> > >
{
    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith f (ReaderT m) (ReaderT n) = ReaderT $ \ a ->
    //   mzipWith f (m a) (n a)
    template<typename C, typename A, typename B>
    static ReaderT<R, _M, C> mzipWith(function_t<C(A, B)> const& f,
        ReaderT<R, _M, fdecay<A> > const& m, ReaderT<R, _M, fdecay<B> > const& n)
    {
        return _([f, m, n](R const& r) { return MonadZip<_M>::mzipWith(f, m.run(r), n.run(r)); });
    }
};

// MonadTrans
template<typename R, typename _M>
struct MonadTrans<_ReaderT<R, _M> >
{
    // lift = liftReaderT
    template<typename MA>
    static typename std::enable_if<std::is_same<_M, base_class_t<MA> >::value,
        ReaderT<R, _M, value_type_t<MA> >
    >::type lift(MA const& m){
        return liftReaderT<R>(m);
    }
};

// MonadReader
template<typename R, typename _M>
struct MonadReader<R, _ReaderT<R, _M> > : _MonadReader<R, _ReaderT<R, _M>, MonadReader<R, _ReaderT<R, _M> > >
{
    using base_class = _ReaderT<R, _M>;
    using super = _MonadReader<R, base_class, MonadReader>;

    template<typename T>
    using type = typename super::template type<T>;

    // ask = ReaderT.ask
    static type<R> ask() {
        return base_class::ask();
    }

    // local = ReaderT.local
    template<typename A, typename RArg>
    static typename std::enable_if<is_same_as<R, RArg>::value, type<A> >::type
    local(function_t<R(RArg)> const& f, type<A> const& m){
        return base_class::local(f, m);
    }

    // reader = ReaderT.reader
    template<typename A, typename RArg>
    static typename std::enable_if<is_same_as<R, RArg>::value, type<A> >::type
    reader(function_t<A(RArg)> const& f){
        return base_class::reader(f);
    }
};

// MonadWriter
template<typename W, typename R, typename _M>
struct MonadWriter<W, _ReaderT<R, _M> > : _MonadWriter<W, _ReaderT<R, _M>, MonadWriter<W, _ReaderT<R, _M> > >
{
    using base_class = _ReaderT<R, _M>;
    using super = _MonadWriter<W, base_class, MonadWriter>;

    template<typename T>
    using type = typename super::template type<T>;

    // writer = lift . writer
    template<typename A>
    static type<A> writer(pair_t<A, W> const& p) {
        return MonadTrans<base_class>::lift(MonadWriter<W, _M>::writer(f));
    }

    // tell = lift . tell
    static type<None> tell(W const& w) {
        return MonadTrans<base_class>::lift(MonadWriter<W, _M>::tell(w));
    }

    // listen = mapReaderT listen
    template<typename A>
    static type<pair_t<A, W> > listen(ReaderT<R, _M, A> const& m) {
        return mapReaderT(_(MonadWriter<W, _M>::template listen<A>), m);
    }

    // pass = mapReaderT pass 
    template<typename A>
    static type<A> pass(ReaderT<R, _M, pair_t<A, function_t<W(W const&)> > > const& m) {
        return mapReaderT(_(MonadWriter<W, _M>::template pass<A>), m);
    }
};

// MonadState
template<typename S, typename R, typename _M>
struct MonadState<S, _ReaderT<R, _M> > : _MonadState<S, _ReaderT<R, _M>, MonadState<S, _ReaderT<R, _M> > >
{
    using base_class = _ReaderT<R, _M>;
    using super = _MonadState<S, base_class, MonadState>;

    template<typename T>
    using type = typename super::template type<T>;

    // state = lift . state
    template<typename A>
    static type<A> state(function_t<pair_t<A, S>(S const&)> const& f){
        return MonadTrans<base_class>::lift(MonadState<S, _M>::state(f));
    }
    
    // get = lift get
    static type<S> get() {
        return MonadTrans<base_class>::lift(MonadState<S, _M>::get());
    }

    // put = lift . put
    static type<None> put(S const& s) {
        const ReaderT<R, _M, None> rrr = MonadTrans<base_class>::lift(MonadState<S, _M>::put(s));
        return rrr;
    }
};

_FUNCPROG_END
