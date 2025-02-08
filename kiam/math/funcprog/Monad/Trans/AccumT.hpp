#pragma once

#include "ReaderT.hpp"
#include "WriterT.hpp"
#include "StateT.hpp"

_FUNCPROG_BEGIN

/*
-----------------------------------------------------------------------------
-- The lazy 'AccumT' monad transformer, which adds accumulation
-- capabilities (such as declarations or document patches) to a given monad.
--
-- This monad transformer provides append-only accumulation
-- during the computation. For more general access, use
-- "Control.Monad.Trans.State" instead.
-----------------------------------------------------------------------------
*/
template<typename W, typename _M>
struct _AccumT;

template<typename W>
using _Accum = _AccumT<W, _Identity>;

template<typename W, typename _M, typename A>
struct AccumT;

#define ACCUMT_(W, _M, A) BOOST_IDENTITY_TYPE((AccumT<W, _M, A>))
#define ACCUMT(W, _M, A) typename ACCUMT_(W, _M, A)

/*
-- ---------------------------------------------------------------------------
-- | An accumulation monad parameterized by the type @w@ of output to accumulate.
--
-- The 'return' function produces the output 'mempty', while @>>=@
-- combines the outputs of the subcomputations using 'mappend'.
*/
template<typename W, typename A>
using Accum = AccumT<W, _Identity, A>;

#define ACCUM_(W, A) BOOST_IDENTITY_TYPE((Accum<W, A>))
#define ACCUM(W, A) typename ACCUM_(W, A)

/*
-- | Construct an accumulation computation from a (result, output) pair.
-- (The inverse of 'runAccum'.)
accum :: (Monad m) => (w -> (a, w)) -> AccumT w m a
accum f = AccumT $ \ w -> return (f w)
*/
template<typename _M, typename W, typename A>
constexpr AccumT<W, _M, A> accum(function_t<pair_t<A, W>(W const&)> const& f){
    return _([f](W const& w){ return Monad<_M>::mreturn(f(w)); });
}

template<typename W>
struct __AccumT
{
    template<typename _M>
    using base_type = _AccumT<W, _M>;
};

template<typename W, typename _M>
struct _AccumT : __AccumT<W>
{
    static_assert(_is_monad_v<_M>, "Should be a monad");

    using base_class = _AccumT;

    template<typename A>
    using type = AccumT<W, _M, A>;

    /*
    -- | @'look'@ is an action that fetches all the previously accumulated output.
    look :: (Monoid w, Monad m) => AccumT w m w
    look = AccumT $ \ w -> return (w, mempty)
    */
    static constexpr type<W> look(){
        return _([](W const& w){
            return Monad<_M>::mreturn(make_pair_t(w, W::mempty()));
        });
    }

    /*
    -- | @'add' w@ is an action that produces the output @w@.
    add :: (Monad m) => w -> AccumT w m ()
    add w = accum $ const ((), w)
    */
    static constexpr type<None> add(W const& w){
        return accum<_M>(
            _const_<W>(pair_t<None, W>(None(), w)));
    }

    /*
    -- | @'look'@ is an action that retrieves a function of the previously accumulated output.
    looks :: (Monoid w, Monad m) => (w -> a) -> AccumT w m a
    looks f = AccumT $ \ w -> return (f w, mempty)
    */
    template<typename A, typename... Args>
    static constexpr auto looks(function_t<A(W const&, Args...)> const& f){
        return _([f](W const& w){
            return Monad<_M>::mreturn(
                pair_t<remove_f0_t<function_t<A(Args...)> >, W>(invoke_f0(f << w), W::mempty()));
        });
    }

    /*
    -- | Lift a @listen@ operation to the new monad.
    liftListen :: (Monad m) => Listen w m (a, s) -> Listen w (AccumT s m) a
    liftListen listen m = AccumT $ \ s -> do
        ~((a, s'), w) <- listen (runAccumT m s)
        return ((a, w), s')
    */
    template<typename S, typename A>
    static constexpr auto liftListen(Listen<W, _M, pair_t<A, S> > const& listen){
        return _([listen](AccumT<S, _M, A> const& m){
            return _([listen, m](S const& s){
                return _do(p, m.run(s),
                    return Monad<_M>::mreturn(pair_t<pair_t<A, W>, S>(pair_t<A, S>(fst(fst(p)), snd(p)), snd(fst(p)))););
                });
            });
    }
    /*
    -- | Lift a @pass@ operation to the new monad.
    liftPass :: (Monad m) => Pass w m (a, s) -> Pass w (AccumT s m) a
    liftPass pass m = AccumT $ \ s -> pass $ do
        ~((a, f), s') <- runAccumT m s
        return ((a, s'), f)
    */
    template<typename S, typename A>
    static constexpr auto liftPass(Pass<W, _M, pair_t<A, S> > const& pass){
        return _([pass](AccumT<S, _M, A> const& m){
            return _([pass, m](S const& s){
                return pass(_do(p, m.run(s),
                    return Monad<_M>::mreturn(pair_t<pair_t<A, S>, W>(pair_t<A, S>(fst(fst(p)), snd(p)), snd(fst(p))));));
                });
            });
    }
};

template<typename W, typename _M, typename A>
struct AccumT : _AccumT<W, _M>
{
    using super = _AccumT<W, _M>;

    template<typename T>
    using M_type = typename _M::template type<T>;

    using value_type = A;
    using return_type = M_type<pair_t<A, W> >;
    using function_type = function_t<return_type(W const&)>;

    AccumT(function_type const& func) : func(func){}

    /*
    -- | Unwrap an accumulation computation.
    runAccumT :: AccumT w m a -> w -> m (a, w)
    runAccumT (AccumT f) = f
    */
    constexpr return_type run(W const& w) const {
        return func(w);
    }

    /*
    -- | Evaluate an accumulation computation with the given initial output history
    -- and return the final value, discarding the final output.
    --
    -- * @'evalAccumT' m w = 'liftM' 'fst' ('runAccumT' m w)@
    evalAccumT :: (Monad m, Monoid w) => AccumT w m a -> w -> m a
    evalAccumT m w = do
        ~(a, _) <- runAccumT m w
        return a
    */
    constexpr auto eval(W const& w) const {
        return _do(p, run(w), return Monad<_M>::mreturn(fst(p)););
    }

    /*
    -- | Extract the output from an accumulation computation.
    --
    -- * @'execAccumT' m w = 'liftM' 'snd' ('runAccumT' m w)@
    execAccumT :: (Monad m) => AccumT w m a -> w -> m w
    execAccumT m w = do
        ~(_, w') <- runAccumT m w
        return w'
    */
    constexpr auto exec(W const& w) const {
        return _do(p, run(w), return Monad<_M>::mreturn(snd(p)););
    }

private:
    const function_type func;
};

template<typename W, typename _M, typename A>
constexpr auto runAccumT(AccumT<W, _M, A> const& m, W const& w){
    return m.run(w);
}

template<typename W, typename _M, typename A>
constexpr auto _runAccumT(AccumT<W, _M, A> const& m){
    return _([m](W const& w){ return runAccumT(m, w); });
}

template<typename W, typename _M, typename A>
constexpr auto evalAccumT(AccumT<W, _M, A> const& m, W const& w){
    return m.eval(w);
}

template<typename W, typename _M, typename A>
constexpr auto _evalAccumT(AccumT<W, _M, A> const& m){
    return _([m](W const& w){ return evalAccumT(m, w); });
}

/*
-- | Evaluate a accum computation with the given initial accum
-- and return the final accum, discarding the final value.
--
-- * @'execAccumT' m w = 'liftM' 'snd' ('runAccumT' m w)@
execAccumT :: (Monad m) => AccumT w m a -> w -> m w
execAccumT m w = do
    ~(_, w') <- runAccumT m w
    return w'
*/
template<typename W, typename _M, typename A>
constexpr auto execAccumT(AccumT<W, _M, A> const& m, W const& w){
    return m.exec(w);
}

template<typename W, typename _M, typename A>
constexpr auto _execAccumT(AccumT<W, _M, A> const& m){
    return _([m](W const& w){ return execAccumT(m, w); });
}

/*
-- | Map both the return value and output of a computation using
-- the given function.
--
-- * @'runAccumT' ('mapAccumT' f m) = f . 'runAccumT' m@
mapAccumT :: (m (a, w) -> n (b, w)) -> AccumT w m a -> AccumT w n b
mapAccumT f m = AccumT (f . runAccumT m)
*/
template<typename W, typename _M, typename A, typename NB, typename Arg>
static constexpr std::enable_if_t<
    std::is_same_v<Arg, typename _M::template type<pair_t<A, W> > > &&
    is_pair_v<value_type_t<NB> > &&
    std::is_same_v<W, snd_type_t<value_type_t<NB> > >,
    AccumT<W, base_class_t<NB>, fst_type_t<value_type_t<NB> > >
> mapAccumT(function_t<NB(Arg const&)> const& f, AccumT<W, _M, A> const& m){
    return f & _runAccumT(m);
}

template<typename W, typename _M, typename A, typename NB, typename Arg>
static constexpr std::enable_if_t<
    std::is_same_v<Arg, typename _M::template type<pair_t<A, W> > > &&
    is_pair_v<value_type_t<NB> > &&
    std::is_same_v<W, snd_type_t<value_type_t<NB> > >,
    function_t<AccumT<W, base_class_t<NB>, fst_type_t<value_type_t<NB> > >(AccumT<W, _M, A> const&)>
> _mapAccumT(function_t<NB(Arg const&)> const& f){
    return _([f](AccumT<W, _M, A> const& m){
        return mapAccumT(f, m);
    });
}

// Accum
/*
-- | Unwrap an accumulation computation as a (result, output) pair.
-- (The inverse of 'accum'.)
runAccum :: Accum w a -> w -> (a, w)
runAccum m = runIdentity . runAccumT m
*/
template<typename W, typename A>
constexpr auto runAccum(Accum<W, A> const& m, W const& w){
    return (_(runIdentity<pair_t<A, W> >) & _runAccumT(m))(w);
}

template<typename W, typename A>
constexpr auto _runAccum(Accum<W, A> const& m){
    return _([m](W const& w){ return runAccum(m, w); });
}

/*
-- | Evaluate an accumulation computation with the given initial output history
-- and return the final value, discarding the final output.
--
-- * @'evalAccum' m w = 'fst' ('runAccum' m w)@
evalAccum :: (Monoid w) => Accum w a -> w -> a
evalAccum m w = fst (runAccum m w)
*/
template<typename W, typename A>
constexpr auto evalAccum(Accum<W, A> const& m, W const& w){
    return fst(runAccum(m, w));
}

template<typename W, typename A>
constexpr auto _evalAccum(Accum<W, A> const& m){
    return _([m](W const& w){ return evalAccum(m, w); });
}
/*
-- | Extract the output from an accumulation computation.
--
-- * @'execAccum' m w = 'snd' ('runAccum' m w)@
execAccum :: Accum w a -> w -> w
execAccum m w = snd (runAccum m w)
*/
template<typename W, typename A>
constexpr auto execAccum(Accum<W, A> const& m, W const& w){
    return snd(runAccum(m, w));
}

template<typename W, typename A>
constexpr auto _execAccum(Accum<W, A> const& m){
    return _([m](W const& w){ return execAccum(m, w); });
}

/*
-- | Map both the return value and output of a computation using
-- the given function.
--
-- * @'runAccum' ('mapAccum' f m) = f . 'runAccum' m@
mapAccum :: ((a, w) -> (b, w)) -> Accum w a -> Accum w b
mapAccum f = mapAccumT (Identity . f . runIdentity)
*/
template<typename W, typename A, typename B>
constexpr auto mapAccum(function_t<pair_t<B, W>(pair_t<A, W> const&)> const& f, Accum<W, A> const& m){
    return mapAccumT(_(Identity_<B>) & f & _(runIdentity<pair_t<A, W> >), m);
}

template<typename W, typename A, typename B>
constexpr auto _mapAccum(function_t<pair_t<B, W>(pair_t<A, W> const&)> const& f){
    return _([f](Accum<W, A> const& m){ return mapAccum(f, m); });
}

// Functor
template<typename W, typename _M>
struct _is_functor<_AccumT<W, _M> > : _is_functor<_M> {};

template<typename W, typename _M, typename A>
struct is_functor<AccumT<W, _M, A> > : _is_functor<_M> {};

template<typename W, typename _M>
struct Functor<_AccumT<W, _M> > : _Functor<_AccumT<W, _M> >
{
    // fmap f = mapAccumT $ fmap $ \ ~(a, w) -> (f a, w)
    template<typename Ret, typename Arg, typename... Args>
    static constexpr AccumT<W, _M, remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, AccumT<W, _M, fdecay<Arg> > const& m){
        using A = fdecay<Arg>;
        return mapAccumT(_fmap<typename _M::template type<pair_t<A, W> > >(_([f](pair_t<A, W> const& p){
            return pair_t<remove_f0_t<function_t<Ret(Args...)> >, W>(invoke_f0(f << fst(p)), snd(p));
        })), m);
    }
};

// Applicative
template<typename W, class _M>
struct _is_applicative<_AccumT<W, _M> > : std::integral_constant<bool, is_monoid_v<W> > {};

template<typename W, class _M, typename A>
struct is_applicative<AccumT<W, _M, A> > : std::integral_constant<bool, is_monoid_v<W> > {};

template<typename W, typename _M>
struct Applicative<_AccumT<W, _M> > : Functor<_AccumT<W, _M> >, _Applicative<_AccumT<W, _M> >
{
    using super = Functor<_AccumT<W, _M> >;

    // pure a  = AccumT $ const $ return (a, mempty)
    template<typename A>
    static constexpr auto pure(A const& a){
        return _const_<W>(Applicative<_M>::pure(make_pair_t(a, W::mempty())));
    }

    /*
    AccumT mf <*> AccumT mx = AccumT $ \ w -> do
        ~(f, w') <- mf w
        ~(x, w'') <- mx w'
        return (f x, w'')
    mf <*> mv = AccumT $ \ w -> do
      ~(f, w')  <- runAccumT mf w
      ~(v, w'') <- runAccumT mv (w `mappend` w')
      return (f v, w' `mappend` w'')
    */
    template<typename Ret, typename Arg, typename... Args>
    static constexpr AccumT<W, _M, remove_f0_t<function_t<Ret(Args...)> > >
    apply(AccumT<W, _M, function_t<Ret(Arg, Args...)> > const& mf, AccumT<W, _M, fdecay<Arg> > const& m){
        return _([mf, m](W const& w){
            return _do2(pf, mf.run(w), pm, m.run(Monoid_t<W>::mappend(w, snd(pf))),
                const function_t<Ret(Arg, Args...)> f = fst(pf);
                return Monad<_M>::mreturn(pair_t<remove_f0_t<function_t<Ret(Args...)> >, W>(
                    invoke_f0(f << fst(pm)), Monoid_t<W>::mappend(snd(pf), snd(pm))));
                );
            });
    }
};

// Monad
template<typename W, class _M>
struct _is_monad<_AccumT<W, _M> > : std::integral_constant<bool, is_monoid_v<W> > {};

template<typename W, class _M, typename A>
struct is_monad<AccumT<W, _M, A> > : std::integral_constant<bool, is_monoid_v<W> > {};

template<typename T>
struct is_accum_monad : std::false_type {};

template<typename W, typename _M, typename A>
struct is_accum_monad<AccumT<W, _M, A> > : std::true_type {};

template<typename W, typename _M>
struct Monad<_AccumT<W, _M> > : Applicative<_AccumT<W, _M> >, _Monad<_AccumT<W, _M> >
{
    using super = Applicative<_AccumT<W, _M> >;

    // return a  = AccumT $ const $ return (a, mempty)
    template<typename A>
    static constexpr AccumT<W, _M, A> mreturn(A const& a){
        return _const_<W>(Monad<_M>::mreturn(make_pair_t(a, W::mempty())));
    }

    // m >>= k  = AccumT $ \ w -> do
    //    ~(a, w')  <- runAccumT m w
    //    ~(b, w'') <- runAccumT (k a) (w `mappend` w')
    //    return (b, w' `mappend` w'')
    template<typename Ret, typename Arg, typename... Args>
    static constexpr remove_f0_t<function_t<AccumT<W, _M, Ret>(Args...)> >
    mbind(AccumT<W, _M, fdecay<Arg> > const& m, function_t<AccumT<W, _M, Ret>(Arg, Args...)> const& f){
        return invoke_f0(_([m, f](Args... args){
            return _([m, f, args...](W const& w){
                using pair_type = pair_t<fdecay<Arg>, W>;
                return _do2(pa, m.run(w),
                    pb, f(fst(pa), args...).run(Monoid_t<W>::mappend(w, snd(pa))),
                    return pair_type(fst(pb), Monoid_t<W>::mappend(snd(pa), snd(pb))););
                });
        }));
    }
};

// MonadPlus
template<typename W, typename _M>
struct _is_monad_plus<_AccumT<W, _M> > : _is_monad_plus<_M> {};

template<typename W, typename _M>
struct MonadPlus<_AccumT<W, _M> > : Monad<_AccumT<W, _M> >, _MonadPlus<_AccumT<W, _M> >
{
    using super = Monad<_AccumT<W, _M> >;

    // mzero = AccumT $ const mzero
    template<typename A>
    static constexpr AccumT<W, _M, A> mzero(){
        return _const_<W>(MonadPlus<_M>::mzero());
    }

    // m `mplus` n = AccumT $ \ w -> runAccumT m w `mplus` runAccumT n w
    template<typename A>
    static constexpr auto mplus(AccumT<W, _M, A> const& m, AccumT<W, _M, A> const& n){
        return _([m, n](W const& w){ return MonadPlus<_M>::mplus(m.run(w), n.run(w)); });
    }
};

// Alternative
template<typename W, typename _M>
struct _is_alternative<_AccumT<W, _M> > : _is_monad_plus<_M> {};

template<typename W, typename _M>
struct Alternative<_AccumT<W, _M> > : _Alternative<_AccumT<W, _M> >
{
    // empty   = AccumT $ const mzero
    template<typename A>
    static constexpr auto empty(){
        return _const_<W>(MonadPlus<_M>::mzero());
    }

    // m <|> n = AccumT $ \ w -> runAccumT m w `mplus` runAccumT n w
    template<typename A>
    static constexpr auto alt_op(AccumT<W, _M, A> const& m, AccumT<W, _M, A> const& n){
        return _([m, n](W const& w){ return MonadPlus<_M>::mplus(m.run(w), n.run(w)); });
    }
};

// MonadTrans
template<typename W>
struct MonadTrans<__AccumT<W> >
{
    // lift m = AccumT $ const $ do
    //    a <- m
    //    return (a, mempty)
    template<typename M>
    static constexpr monad_type<M, AccumT<W, base_class_t<M>, value_type_t<M> > > lift(M const& m){
        return _const_<W>(_do(a, m,
            return Monad_t<M>::mreturn(pair_t<value_type_t<M>, W>(a, Monoid_t<W>::mempty()));));
    }
};

/*
-- | Convert a read-only computation into an accumulation computation.
readerToAccumT :: (Functor m, Monoid w) => ReaderT w m a -> AccumT w m a
readerToAccumT (ReaderT f) = AccumT $ \ w -> fmap (\ a -> (a, mempty)) (f w)
*/
template<typename W, typename _M, typename A>
constexpr auto readerToAccumT(ReaderT<W, _M, A> const& rdr){
    return _([rdr](W const& w){
        return fmap(_([](A const& a){ return pair_t<A, W>(a, Monoid_t<W>::mempty()); }), rdr.run(w));
    });
}

/*
-- | Convert a writer computation into an accumulation computation.
writerToAccumT :: WriterT w m a -> AccumT w m a
writerToAccumT (WriterT m) = AccumT $ const $ m
*/
template<typename W, typename _M, typename A>
constexpr auto writerToAccumT(WriterT<W, _M, A> const& wrt){
    return _const_<W>(wrt.run());
}

/*
-- | Convert an accumulation (append-only) computation into a fully
-- stateful computation.
accumToStateT :: (Functor m, Monoid s) => AccumT s m a -> StateT s m a
accumToStateT (AccumT f) =
    StateT $ \ w -> fmap (\ ~(a, w') -> (a, w `mappend` w')) (f w)
*/
template<typename S, typename _M, typename A>
constexpr auto accumToStateT(AccumT<S, _M, A> const& acc){
    return _([acc](S const& w){
        return fmap(_([w](pair_t<A, S> const& p){ return pair_t<A, S>(fst(p), Monoid_t<S>::mappend(w, snd(p))); }), acc.run(w));
    });
}

_FUNCPROG_END
