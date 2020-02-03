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
AccumT<W, _M, A> accum(function_t<pair_t<A, W>(W const&)> const& f) {
    return _([f](W const& w) { return Monad<_M>::mreturn(f(w)); });
}

template<typename W>
struct __AccumT
{
    template<typename _M>
    using mt_type = _AccumT<W, _M>;
};

template<typename W, typename _M>
struct _AccumT
{
    static_assert(is_monad<_M>::value, "Should be a monad");

    using base_class = _AccumT;

    template<typename A>
    using type = AccumT<W, _M, A>;

    /*
    -- | @'look'@ is an action that fetches all the previously accumulated output.
    look :: (Monoid w, Monad m) => AccumT w m w
    look = AccumT $ \ w -> return (w, mempty)
    */
    static type<W> look() {
        return _([](W const& w) {
            return Monad<_M>::mreturn(pair_t<W, W>(w, Monoid_t<W>::template mempty<value_type_t<W> >()));
        });
    }

    /*
    -- | @'add' w@ is an action that produces the output @w@.
    add :: (Monad m) => w -> AccumT w m ()
    add w = accum $ const ((), w)
    */
    static type<None> add(W const& w) {
        return accum<_M>(
            _const_<W>(pair_t<None, W>(None(), w)));
    }

    /*
    -- | @'look'@ is an action that retrieves a function of the previously accumulated output.
    looks :: (Monoid w, Monad m) => (w -> a) -> AccumT w m a
    looks f = AccumT $ \ w -> return (f w, mempty)
    */
    template<typename A, typename... Args>
    static type<remove_f0_t<function_t<A(Args...)> > > looks(function_t<A(W const&, Args...)> const& f) {
        return _([f](W const& w) {
            return Monad<_M>::mreturn(
                pair_t<remove_f0_t<function_t<A(Args...)> >, W>(invoke_f0(f << w), Monoid_t<W>::template mempty<value_type_t<W> >()));
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
    static Listen<W, _AccumT<S, _M>, A> liftListen(Listen<W, _M, pair_t<A, S> > const& listen){
        return _([listen](AccumT<S, _M, A> const& m) {
            return _([listen, m](S const& s) {
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
    static Pass<W, _AccumT<W, _M>, A> liftPass(Pass<W, _M, pair_t<A, S> > const& pass){
        return _([pass](AccumT<S, _M, A> const& m) {
            return _([pass, m](S const& s) {
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

    AccumT(function_type const& func) : func(func) {}

    /*
    -- | Unwrap an accumulation computation.
    runAccumT :: AccumT w m a -> w -> m (a, w)
    runAccumT (AccumT f) = f
    */
    return_type run(W const& w) const {
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
    M_type<A> eval(W const& w) const {
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
    M_type<W> exec(W const& w) const {
        return _do(p, run(w), return Monad<_M>::mreturn(snd(p)););
    }

private:
    const function_type func;
};

template<typename W, typename _M, typename A>
typename _M::template type<pair_t<A, W> >
runAccumT(AccumT<W, _M, A> const& m, W const& w) {
    return m.run(w);
}

template<typename W, typename _M, typename A>
function_t<typename _M::template type<pair_t<A, W> >(W const&)>
_runAccumT(AccumT<W, _M, A> const& m) {
    return [m](W const& w) { return runAccumT(m, w); };
}

template<typename W, typename _M, typename A>
typename _M::template type<A>
evalAccumT(AccumT<W, _M, A> const& m, W const& w) {
    return m.eval(w);
}

template<typename W, typename _M, typename A>
function_t<typename _M::template type<A>(W const&)>
_evalAccumT(AccumT<W, _M, A> const& m) {
    return [m](W const& w) { return evalAccumT(m, w); };
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
typename _M::template type<W>
execAccumT(AccumT<W, _M, A> const& m, W const& w) {
    return m.exec(w);
}

template<typename W, typename _M, typename A>
function_t<typename _M::template type<W>(W const&)>
_execAccumT(AccumT<W, _M, A> const& m) {
    return [m](W const& w) { return execAccumT(m, w); };
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
static typename std::enable_if<
    std::is_same<Arg, typename _M::template type<pair_t<A, W> > >::value &&
    is_pair<value_type_t<NB> >::value &&
    std::is_same<W, snd_type_t<value_type_t<NB> > >::value,
    AccumT<W, base_class_t<NB>, fst_type_t<value_type_t<NB> > >
>::type mapAccumT(function_t<NB(Arg const&)> const& f, AccumT<W, _M, A> const& m) {
    return f & _runAccumT(m);
}

template<typename W, typename _M, typename A, typename NB, typename Arg>
static typename std::enable_if<
    std::is_same<Arg, typename _M::template type<pair_t<A, W> > >::value &&
    is_pair<value_type_t<NB> >::value &&
    std::is_same<W, snd_type_t<value_type_t<NB> > >::value,
    function_t<AccumT<W, base_class_t<NB>, fst_type_t<value_type_t<NB> > >(AccumT<W, _M, A> const&)>
>::type _mapAccumT(function_t<NB(Arg const&)> const& f) {
    return [f](AccumT<W, _M, A> const& m) {
        return mapAccumT(f, m);
    };
}

// Accum
/*
-- | Unwrap an accumulation computation as a (result, output) pair.
-- (The inverse of 'accum'.)
runAccum :: Accum w a -> w -> (a, w)
runAccum m = runIdentity . runAccumT m
*/
template<typename W, typename A>
pair_t<A, W> runAccum(Accum<W, A> const& m, W const& w) {
    return (_(runIdentity<pair_t<A, W> >) & _runAccumT(m))(w);
}

template<typename W, typename A>
function_t<pair_t<A, W>(W const&)>
_runAccum(Accum<W, A> const& m) {
    return [m](W const& w) { return runAccum(m, w); };
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
A evalAccum(Accum<W, A> const& m, W const& w) {
    return fst(runAccum(m, w));
}

template<typename W, typename A>
function_t<A(W const&)> _evalAccum(Accum<W, A> const& m) {
    return [m](W const& w) { return evalAccum(m, w); };
}
/*
-- | Extract the output from an accumulation computation.
--
-- * @'execAccum' m w = 'snd' ('runAccum' m w)@
execAccum :: Accum w a -> w -> w
execAccum m w = snd (runAccum m w)
*/
template<typename W, typename A>
W execAccum(Accum<W, A> const& m, W const& w) {
    return snd(runAccum(m, w));
}

template<typename W, typename A>
function_t<W(W const&)> _execAccum(Accum<W, A> const& m) {
    return [m](W const& w) { return execAccum(m, w); };
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
Accum<W, B> mapAccum(function_t<pair_t<B, W>(pair_t<A, W> const&)> const& f, Accum<W, A> const& m) {
    return mapAccumT(_(Identity_<B>) & f & _(runIdentity<pair_t<A, W> >), m);
}

template<typename W, typename A, typename B>
function_t<Accum<W, B>(Accum<W, A> const&)>
_mapAccum(function_t<pair_t<B, W>(pair_t<A, W> const&)> const& f) {
    return [f](Accum<W, A> const& m) { return mapAccum(f, m); };
}

// Functor
template<typename W, typename _M>
struct is_functor<_AccumT<W, _M> > : is_functor<_M> {};

template<typename W, typename _M>
struct is_same_functor<_AccumT<W, _M>, _AccumT<W, _M> > : std::true_type {};

template<typename W, typename _M>
struct Functor<_AccumT<W, _M> >
{
    // fmap f = mapAccumT $ fmap $ \ ~(a, w) -> (f a, w)
    template<typename A, typename Ret, typename Arg, typename... Args>
    static typename std::enable_if<is_same_as<A, Arg>::value,
        AccumT<W, _M, remove_f0_t<function_t<Ret(Args...)> > >
    >::type fmap(function_t<Ret(Arg, Args...)> const& f, AccumT<W, _M, A> const& m) {
        return mapAccumT(_fmap<typename _M::template type<pair_t<A, W> > >(_([f](pair_t<A, W> const& p) {
            return pair_t<remove_f0_t<function_t<Ret(Args...)> >, W>(invoke_f0(f << fst(p)), snd(p));
        })), m);
    }
};

// Applicative
template<typename W, class _M>
struct is_applicative<_AccumT<W, _M> > : std::integral_constant<bool, is_monoid_t<W>::value> {};

template<typename W, class _M>
struct is_same_applicative<_AccumT<W, _M>, _AccumT<W, _M> > : std::integral_constant<bool, is_monoid_t<W>::value> {};

template<typename W, typename _M>
struct Applicative<_AccumT<W, _M> > : Functor<_AccumT<W, _M> >
{
    using super = Functor<_AccumT<W, _M> >;

    // pure a  = AccumT $ const $ return (a, mempty)
    template<typename A>
    static AccumT<W, _M, A> pure(A const& a) {
        return _const_<W>(Monad<_M>::mreturn(pair_t<A, W>(a, Monoid_t<W>::template mempty<value_type_t<W> >())));
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
    static AccumT<W, _M, remove_f0_t<function_t<Ret(Args...)> > >
    apply(AccumT<W, _M, function_t<Ret(Arg, Args...)> > const& mf, AccumT<W, _M, fdecay<Arg> > const& m) {
        return _([mf, m](W const& w) {
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
struct is_monad<_AccumT<W, _M> > : std::integral_constant<bool, is_monoid_t<W>::value> {};

template<typename W, class _M>
struct is_same_monad<_AccumT<W, _M>, _AccumT<W, _M> > : std::integral_constant<bool, is_monoid_t<W>::value> {};

template<typename T>
struct is_accum_monad : std::false_type {};

template<typename W, typename _M, typename A>
struct is_accum_monad<AccumT<W, _M, A> > : std::true_type {};

template<typename W, typename _M>
struct Monad<_AccumT<W, _M> > : Applicative<_AccumT<W, _M> >
{
    using super = Applicative<_AccumT<W, _M> >;

    // return a  = AccumT $ const $ return (a, mempty)
    template<typename A>
    static AccumT<W, _M, A> mreturn(A const& a) {
        return _const_<W>(Monad<_M>::mreturn(pair_t<A, W>(a, Monoid_t<W>::mempty())));
    }

    // m >>= k  = AccumT $ \ w -> do
    //    ~(a, w')  <- runAccumT m w
    //    ~(b, w'') <- runAccumT (k a) (w `mappend` w')
    //    return (b, w' `mappend` w'')
    template<typename Ret, typename Arg, typename... Args>
    static remove_f0_t<function_t<AccumT<W, _M, Ret>(Args...)> >
    mbind(AccumT<W, _M, fdecay<Arg> > const& m, function_t<AccumT<W, _M, Ret>(Arg, Args...)> const& f) {
        return invoke_f0(_([m, f](Args... args) {
            return _([m, f, args...](W const& w) {
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
struct is_monad_plus<_AccumT<W, _M> > : is_monad_plus<_M> {};

template<typename W, typename _M>
struct is_same_monad_plus<_AccumT<W, _M>, _AccumT<W, _M> > : is_monad_plus<_M> {};

template<typename W, typename _M>
struct MonadPlus<_AccumT<W, _M> > : Monad<_AccumT<W, _M> >
{
    using super = Monad<_AccumT<W, _M> >;

    // mzero = AccumT $ const mzero
    template<typename A>
    static AccumT<W, _M, A> mzero() {
        return _const_<W>(MonadPlus<_M>::mzero());
    }

    template<typename T>
    struct mplus_result_type;

    template<typename T>
    using mplus_result_type_t = typename mplus_result_type<T>::type;

    template<typename A>
    struct mplus_result_type<AccumT<W, _M, A> >
    {
        using type = AccumT<W, _M, A>;
    };

    // m `mplus` n = AccumT $ \ w -> runAccumT m w `mplus` runAccumT n w
    template<typename A>
    static AccumT<W, _M, A> mplus(AccumT<W, _M, A> const& m, AccumT<W, _M, A> const& n) {
        return _([m, n](W const& w) { return MonadPlus<_M>::mplus(m.run(w), n.run(w)); });
    }
};

// Alternative
template<typename W, typename _M>
struct is_alternative<_AccumT<W, _M> > : is_monad_plus<_M> {};

template<typename W, typename _M>
struct is_same_alternative<_AccumT<W, _M>, _AccumT<W, _M> > : is_monad_plus<_M> {};

template<typename W, typename _M>
struct Alternative<_AccumT<W, _M> >
{
    // empty   = AccumT $ const mzero
    template<typename A>
    static AccumT<W, _M, A> empty() {
        return _const_<W>(MonadPlus<_M>::mzero());
    }

    template<typename T>
    struct alt_op_result_type;

    template<typename T>
    using alt_op_result_type_t = typename alt_op_result_type<T>::type;

    template<typename A>
    struct alt_op_result_type<AccumT<W, _M, A> >
    {
        using type = AccumT<W, _M, A>;
    };

    // m <|> n = AccumT $ \ w -> runAccumT m w `mplus` runAccumT n w
    template<typename A>
    static AccumT<W, _M, A> alt_op(AccumT<W, _M, A> const& m, AccumT<W, _M, A> const& n) {
        return _([m, n](W const& w) { return MonadPlus<_M>::mplus(m.run(w), n.run(w)); });
    }
};

// MonadTrans
template<typename W, typename _M>
struct MonadTrans<_AccumT<W, _M> >
{
    // lift m = AccumT $ const $ do
    //    a <- m
    //    return (a, mempty)
    template<typename MA>
    static typename std::enable_if<std::is_same<_M, base_class_t<MA> >::value,
        AccumT<W, _M, value_type_t<MA> >
    >::type lift(MA const& m) {
        return _const_<W>(_do(a, m,
            return Monad<_M>::mreturn(pair_t<value_type_t<MA>, W>(a, Monoid_t<W>::mempty()));));
    }
};

/*
-- | Convert a read-only computation into an accumulation computation.
readerToAccumT :: (Functor m, Monoid w) => ReaderT w m a -> AccumT w m a
readerToAccumT (ReaderT f) = AccumT $ \ w -> fmap (\ a -> (a, mempty)) (f w)
*/
template<typename W, typename _M, typename A>
AccumT<W, _M, A> readerToAccumT(ReaderT<W, _M, A> const& rdr) {
    return _([rdr](W const& w) {
        return fmap(_([](A const& a) { return pair_t<A, W>(a, Monoid_t<W>::mempty()); }), rdr.run(w));
    });
}

/*
-- | Convert a writer computation into an accumulation computation.
writerToAccumT :: WriterT w m a -> AccumT w m a
writerToAccumT (WriterT m) = AccumT $ const $ m
*/
template<typename W, typename _M, typename A>
AccumT<W, _M, A> writerToAccumT(WriterT<W, _M, A> const& wrt) {
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
typename std::enable_if<is_monoid_t<S>::value, StateT<S, _M, A> >::type
accumToStateT(AccumT<S, _M, A> const& acc) {
    return _([acc](S const& w) {
        return fmap(_([w](pair_t<A, S> const& p) { return pair_t<A, S>(fst(p), Monoid_t<S>::mappend(w, snd(p))); }), acc.run(w));
    });
}

_FUNCPROG_END
