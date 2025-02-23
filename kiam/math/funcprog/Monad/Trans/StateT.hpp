#pragma once

#include "../../Identity.hpp"
#include "../Signatures.hpp"
#include "../MonadState.hpp"
#include "MonadTrans.hpp"

_FUNCPROG_BEGIN

/*
-- ---------------------------------------------------------------------------
-- | A state monad parameterized by the type @s@ of the state to carry.
--
-- The 'return' function leaves the state unchanged, while @>>=@ uses
-- the final state of the first computation as the initial state of
-- the second.
*/
template<typename S, typename _M>
struct _StateT;

template<typename S>
using _State = _StateT<S, _Identity>;

template<typename S, typename _M, typename A>
struct StateT;

#define STATET_(S, _M, A) BOOST_IDENTITY_TYPE((StateT<S, _M, A>))
#define STATET(S, _M, A) typename STATET_(S, _M, A)

template<typename S, typename A>
using State = StateT<S, _Identity, A>;

#define STATE_(S, A) BOOST_IDENTITY_TYPE((State<S, A>))
#define STATE(S, A) typename STATE_(S, A)

/*
-- | Construct a state monad computation from a function.
-- (The inverse of 'runState'.)
state :: (Monad m)
      => (s -> (a, s))  -- ^pure state transformer
      -> StateT s m a   -- ^equivalent state-passing computation
state f = StateT (return . f)
*/
template<typename _M, typename S, typename A>
constexpr StateT<S, _M, A> state(function_t<pair_t<A, S>(S const&)> const& f){
    return _(Monad<_M>::template mreturn<pair_t<A, S> >) & f;
}

template<typename S>
struct __StateT
{
    template<typename _M>
    using base_type = _StateT<S, _M>;
};

template<typename S, typename _M>
struct _StateT
{
    static_assert(_is_monad_v<_M>, "Should be a monad");

    using base_class = _StateT;

    template<typename A>
    using type = StateT<S, _M, A>;

    using state_type = S;
    using state_value_t = type<state_type>;
    using state_transform_function_t = function_t<state_type(state_type const&)>;

    /*
    -- | Fetch the current value of the state within the monad.
    get :: (Monad m) => StateT s m s
    get = state $ \ s -> (s, s)
    */
    static constexpr type<S> get(){
        return state<_M>(_([](state_type const& s){
            return pair_t<state_type, state_type>(s, s);
        }));
    }

    /*
    -- | @'put' s@ sets the state within the monad to @s@.
    put :: (Monad m) => s -> StateT s m ()
    put s = state $ \ _ -> ((), s)
    */
    static constexpr type<None> put(state_type const& s){
        return state<_M>(_([s](state_type const&){
            return pair_t<None, state_type>(None(), s);
        }));
    }

    /*
    -- | @'modify' f@ is an action that updates the state to the result of
    -- applying @f@ to the current state.
    --
    -- * @'modify' f = 'get' >>= ('put' . f)@
    modify :: (Monad m) => (s -> s) -> StateT s m ()
    modify f = state $ \ s -> ((), f s)
    */
    template<typename SArg>
    static constexpr std::enable_if_t<is_same_as_v<S, SArg>, type<None> >
    modify(function_t<S(SArg)> const& f){
        return state<_M>(_([f](state_type const& s){
            return pair_t<None, state_type>(None(), f(s));
        }));
    }

    /*
    -- | Get a specific component of the state, using a projection function
    -- supplied.
    --
    -- * @'gets' f = 'liftM' f 'get'@
    gets :: (Monad m) => (s -> a) -> StateT s m a
    gets f = state $ \ s -> (f s, s)
    */
    template<typename A, typename... Args>
    static constexpr type<remove_f0_t<function_t<A(Args...)> > >
    gets(function_t<A(S const&, Args...)> const& f){
        return state<_M>([f](S const& s){
            return pair_t<A, S>(invoke_f0(f << s), s);
        });
    }

    /*
    -- | Lift a @listen@ operation to the new monad.
    liftListen :: (Monad m) => Listen w m (a,s) -> Listen w (StateT s m) a
    liftListen listen m = StateT $ \ s -> do
        ~((a, s'), w) <- listen (runStateT m s)
        return ((a, w), s')
    */
    template<typename W, typename A>
    static constexpr Listen<W, _StateT<S, _M>, A>
    liftListen(Listen<W, _M, pair_t<A, S> > const& listen){
        return _([listen](StateT<S, _M, A> const& m){
            return _([listen, m](S const& s){
                return _do(p, m.run(s),
                    return Monad<_M>::mreturn(pair_t<pair_t<A, W>, S>(pair_t<A, W>(fst(fst(p)), snd(p)), snd(fst(p)))););
            });
        });
    }
    /*
    -- | Lift a @pass@ operation to the new monad.
    liftPass :: (Monad m) => Pass w m (a,s) -> Pass w (StateT s m) a
    liftPass pass m = StateT $ \ s -> pass $ do
        ~((a, f), s') <- runStateT m s
        return ((a, s'), f)
    */
    template<typename W, typename A>
    static constexpr Pass<W, _StateT<S, _M>, A>
    liftPass(Pass<W, _M, pair_t<A, S> > const& pass){
        return _([pass](StateT<S, _M, A> const& m){
            return _([pass, m](S const& s){
                return pass(_do(p, m.run(s),
                    return Monad<_M>::mreturn(pair_t<pair_t<A, W>, S>(pair_t<A, W>(fst(fst(p)), snd(p)), snd(fst(p))));));
            });
        });
    }

    template<typename SArg>
    static constexpr std::enable_if_t<is_same_as_v<S, SArg>, type<None> >
    while_(function_t<bool(SArg)> const& test, type<None> const& body){
        return _do(s, get(),
            return test(s) ? modify(_execState(body)) >> while_(test, body) : Monad<_StateT>::mreturn(None()););
    }
};

template<typename S, typename _M, typename A>
struct StateT : _StateT<S, _M>
{
    using super = _StateT<S, _M>;

    template<typename T>
    using M_type = typename _M::template type<T>;

    using state_type = S;
    using value_type = A;
    using result_type = pair_t<value_type, state_type>;

    using return_type = M_type<result_type>;
    using function_type = function_t<return_type(state_type const&)>;

    StateT(function_type const& func) : func(func){}
    StateT(function_t<return_type(state_type)> const& f) : func([f](state_type const& state){ return f(state); }){}

    constexpr return_type run(state_type const& s) const {
        return func(s);
    }

    /*
    -- | Evaluate a state computation with the given initial state
    -- and return the final value, discarding the final state.
    --
    -- * @'evalStateT' m s = 'liftM' 'fst' ('runStateT' m s)@
    evalStateT :: (Monad m) => StateT s m a -> s -> m a
    evalStateT m s = do
        ~(a, _) <- runStateT m s
        return a
    */
    constexpr M_type<A> eval(state_type const& s) const {
        return _do(p, run(s), return Monad<_M>::mreturn(fst(p)););
    }

    /*
    -- | Evaluate a state computation with the given initial state
    -- and return the final state, discarding the final value.
    --
    -- * @'execStateT' m s = 'liftM' 'snd' ('runStateT' m s)@
    execStateT :: (Monad m) => StateT s m a -> s -> m s
    execStateT m s = do
        ~(_, s') <- runStateT m s
        return s'
    */
    constexpr M_type<S> exec(state_type const& s) const {
        return _do(p, run(s), return Monad<_M>::mreturn(snd(p)););
    }

private:
    const function_type func;
};

template<typename S, typename _M, typename A>
constexpr typename _M::template type<pair_t<A, S> >
runStateT(StateT<S, _M, A> const& m, S const& s){
    return m.run(s);
}

template<typename S, typename _M, typename A>
constexpr function_t<typename _M::template type<pair_t<A, S> >(S const&)>
_runStateT(StateT<S, _M, A> const& m){
    return [m](S const& s){ return runStateT(m, s); };
}

template<typename S, typename _M, typename A>
constexpr typename _M::template type<A>
evalStateT(StateT<S, _M, A> const& m, S const& s){
    return m.eval(s);
}

template<typename S, typename _M, typename A>
constexpr function_t<typename _M::template type<A>(S const&)>
_evalStateT(StateT<S, _M, A> const& m){
    return [m](S const& s){ return evalStateT(m, s); };
}

/*
-- | Evaluate a state computation with the given initial state
-- and return the final state, discarding the final value.
--
-- * @'execStateT' m s = 'liftM' 'snd' ('runStateT' m s)@
execStateT :: (Monad m) => StateT s m a -> s -> m s
execStateT m s = do
    ~(_, s') <- runStateT m s
    return s'
*/
template<typename S, typename _M, typename A>
constexpr typename _M::template type<S>
execStateT(StateT<S, _M, A> const& m, S const& s){
    return m.exec(s);
}

template<typename S, typename _M, typename A>
constexpr function_t<typename _M::template type<S>(S const&)>
_execStateT(StateT<S, _M, A> const& m){
    return [m](S const& s){ return execStateT(m, s); };
}

/*
-- | Map both the return value and final state of a computation using
-- the given function.
--
-- * @'runStateT' ('mapStateT' f m) = f . 'runStateT' m@
mapStateT :: (m (a, s) -> n (b, s)) -> StateT s m a -> StateT s n b
mapStateT f m = StateT $ f . runStateT m
*/
template<typename S, typename _M, typename A, typename NB, typename Arg>
static constexpr std::enable_if_t<
    std::is_same_v<Arg, typename _M::template type<pair_t<A, S> > > &&
    is_pair_v<value_type_t<NB> > &&
    std::is_same_v<S, snd_type_t<value_type_t<NB> > >,
    StateT<S, base_class_t<NB>, fst_type_t<value_type_t<NB> > >
> mapStateT(function_t<NB(Arg const&)> const& f, StateT<S, _M, A> const& m){
    return f & _runStateT(m);
}

template<typename S, typename _M, typename A, typename NB, typename Arg>
static constexpr std::enable_if_t<
    std::is_same_v<Arg, typename _M::template type<pair_t<A, S> > > &&
    is_pair_v<value_type_t<NB> > &&
    std::is_same_v<S, snd_type_t<value_type_t<NB> > >,
    function_t<StateT<S, base_class_t<NB>, fst_type_t<value_type_t<NB> > >(StateT<S, _M, A> const&)>
> _mapStateT(function_t<NB(Arg const&)> const& f){
    return [f](StateT<S, _M, A> const& m){
        return mapStateT(f, m);
    };
}

/*
-- | @'withStateT' f m@ executes action @m@ on a state modified by
-- applying @f@.
--
-- * @'withStateT' f m = 'modify' f >> m@
withStateT :: (s -> s) -> StateT s m a -> StateT s m a
withStateT f m = StateT $ runStateT m . f
*/
template<typename S, typename _M, typename SArg, typename A>
constexpr std::enable_if_t<is_same_as_v<S, SArg>, StateT<S, _M, A> >
withStateT(function_t<S(SArg)> const& f, StateT<S, _M, A> const& m){
    return _runStateT(m) & f;
}

template<typename S, typename _M, typename SArg, typename A>
constexpr std::enable_if_t<is_same_as_v<S, SArg>,
    function_t<StateT<S, _M, A>(StateT<fdecay<SArg>, _M, A> const&)>
> _withStateT(function_t<S(SArg)> const& f){
    return [f](StateT<S, _M, A> const& m){ return withStateT(f, m); };
}

// State
/*
-- | Unwrap a state monad computation as a function.
-- (The inverse of 'state'.)
runState :: State s a   -- ^state-passing computation to execute
         -> s           -- ^initial state
         -> (a, s)      -- ^return value and final state
runState m = runIdentity . runStateT m
*/
template<typename S, typename A>
constexpr pair_t<A, S> runState(State<S, A> const& m, S const& s){
    return (_(runIdentity<pair_t<A, S> >) & _runStateT(m))(s);
}

template<typename S, typename A>
constexpr function_t<pair_t<A, S>(S const&)>
_runState(State<S, A> const& m){
    return [m](S const& s){ return runState(m, s); };
}

/*
-- | Evaluate a state computation with the given initial state
-- and return the final value, discarding the final state.
--
-- * @'evalState' m s = 'fst' ('runState' m s)@
evalState :: State s a  -- ^state-passing computation to execute
          -> s          -- ^initial value
          -> a          -- ^return value of the state computation
evalState m s = fst (runState m s)
*/
template<typename S, typename A>
constexpr A evalState(State<S, A> const& m, S const& s){
    return fst(runState(m, s));
}

template<typename S, typename A>
constexpr function_t<A(S const&)> _evalState(State<S, A> const& m){
    return [m](S const& s){ return evalState(m, s); };
}
/*
-- | Evaluate a state computation with the given initial state
-- and return the final state, discarding the final value.
--
-- * @'execState' m s = 'snd' ('runState' m s)@
execState :: State s a  -- ^state-passing computation to execute
          -> s          -- ^initial value
          -> s          -- ^final state
execState m s = snd (runState m s)
*/
template<typename S, typename A>
constexpr S execState(State<S, A> const& m, S const& s){
    return snd(runState(m, s));
}

template<typename S, typename A>
constexpr function_t<S(S const&)> _execState(State<S, A> const& m){
    return [m](S const& s){ return execState(m, s); };
}

/*
-- | Map both the return value and final state of a computation using
-- the given function.
--
-- * @'runState' ('mapState' f m) = f . 'runState' m@
mapState :: ((a, s) -> (b, s)) -> State s a -> State s b
mapState f = mapStateT (Identity . f . runIdentity)
*/
template<typename S, typename A, typename B>
constexpr State<S, B> mapState(function_t<pair_t<B, S>(pair_t<A, S> const&)> const& f, State<S, A> const& m){
    return mapStateT(_(Identity_<B>) & f & _(runIdentity<pair_t<A, S> >), m);
}

template<typename S, typename A, typename B>
constexpr function_t<State<S, B>(State<S, A> const&)>
_mapState(function_t<pair_t<B, S>(pair_t<A, S> const&)> const& f){
    return [f](State<S, A> const& m){ return mapState(f, m); };
}

/*
-- | @'withState' f m@ executes action @m@ on a state modified by
-- applying @f@.
--
-- * @'withState' f m = 'modify' f >> m@
withState :: (s -> s) -> State s a -> State s a
withState = withStateT
*/
template<typename S, typename SArg, typename A>
constexpr std::enable_if_t<is_same_as_v<S, SArg>, State<S, A> >
withState(function_t<S(SArg)> const& f, State<S, A> const& m){
    return withStateT(f, m);
}

template<typename S, typename SArg, typename A>
std::enable_if_t<
    is_same_as_v<S, SArg>,
    function_t<State<S, A>(State<S, A> const&)>
> _withState(function_t<S(SArg)> const& f){
    return [f](State<S, A> const& m){ return withState(f, m); };
}

// Functor
template<typename S, typename _M>
struct _is_functor<_StateT<S, _M> > : _is_functor<_M> {};

template<typename S, typename _M, typename A>
struct is_functor<StateT<S, _M, A> > : _is_functor<_M> {};

template<typename S, typename _M>
struct Functor<_StateT<S, _M> > : _Functor<_StateT<S, _M> >
{
    // fmap f m = StateT $ \ s ->
    //   fmap (\ ~(a, s') -> (f a, s')) $ runStateT m s
    template<typename A, typename Ret, typename Arg, typename... Args>
    static constexpr std::enable_if_t<is_same_as_v<A, Arg>,
        StateT<S, _M, remove_f0_t<function_t<Ret(Args...)> > >
    > fmap(function_t<Ret(Arg, Args...)> const& f, StateT<S, _M, A> const& m){
        return _([f, m](S const& s){
            return _([f](pair_t<A, S> const& p){
                return pair_t<remove_f0_t<function_t<Ret(Args...)> >, S>(
                    invoke_f0(f << fst(p)), snd(p));
            }) / m.run(s);
        });
    }
};

// Applicative
template<typename S, typename _M>
struct _is_applicative<_StateT<S, _M> > : _is_applicative<_M> {};

template<typename S, typename _M, typename A>
struct is_applicative<StateT<S, _M, A> > : _is_applicative<_M> {};

template<typename S, typename _M>
struct Applicative<_StateT<S, _M> > : Functor<_StateT<S, _M> >, _Applicative<_StateT<S, _M> >
{
    using super = Functor<_StateT<S, _M> >;

    // pure a = StateT $ \ s -> return (a, s)
    template<typename A>
    static constexpr StateT<S, _M, A> pure(A const& a){
        return _([a](S const& s){ return Applicative<_M>::pure(make_pair_t(a, s)); });
    }

    /*
    StateT mf <*> StateT mx = StateT $ \ s -> do
        ~(f, s') <- mf s
        ~(x, s'') <- mx s'
        return (f x, s'')
    */
    template<typename Ret, typename Arg, typename... Args>
    static constexpr StateT<S, _M, remove_f0_t<function_t<Ret(Args...)> > >
    apply(StateT<S, _M, function_t<Ret(Arg, Args...)> > const& mf, StateT<S, _M, fdecay<Arg> > const& m){
        return _([mf, m](S const& s){
            return _do2(pf, mf.run(s), pm, m.run(snd(pf)),
                const function_t<Ret(Arg, Args...)> f = fst(pf);
                return Applicative<_M>::pure(
                    pair_t<remove_f0_t<function_t<Ret(Args...)> >, S>(invoke_f0(f << fst(pm)), snd(pm)));
            );
        });
    }
};

// m *> k = m >>= \_ -> k
template<typename S, typename _M, typename A, typename B>
constexpr StateT<S, _M, B> operator*=(StateT<S, _M, A> const& a, StateT<S, _M, B> const& b){
    return a >>= _([b](pair_t<A, S> const&){ return b; });
}

// Monad
template<typename S, typename _M>
struct _is_monad<_StateT<S, _M> > : _is_monad<_M> {};

template<typename S, typename _M, typename A>
struct is_monad<StateT<S, _M, A> > : _is_monad<_M> {};

template<typename T>
struct is_state_monad : std::false_type {};

template<typename S, typename _M, typename A>
struct is_state_monad<StateT<S, _M, A> > : std::true_type {};

template<typename S, typename _M>
struct Monad<_StateT<S, _M> > : Applicative<_StateT<S, _M> >, _Monad<_StateT<S, _M> >
{
    using super = Applicative<_StateT<S, _M> >;

    // pure a = StateT $ \ s -> return (a, s)
    template<typename A>
    static constexpr StateT<S, _M, A> mreturn(A const& a){
        return super::pure(a);
    }

    // m >>= k  = StateT $ \ s -> do
    //    ~(a, s') <- runStateT m s
    //    runStateT (k a) s'
    template<typename A, typename Ret, typename Arg, typename... Args>
    static constexpr std::enable_if_t<
        is_same_as_v<A, Arg>,
        remove_f0_t<function_t<StateT<S, _M, Ret>(Args...)> >
    > mbind(StateT<S, _M, A> const& m, function_t<StateT<S, _M, Ret>(Arg, Args...)> const& f){
        return invoke_f0(_([m, f](Args... args){
            return _([m, f, args...](S const& s){
                return _do(p, m.run(s), return f(fst(p), args...).run(snd(p)););
            });
        }));
    }
};

// MonadPlus
template<typename S, typename _M>
struct _is_monad_plus<_StateT<S, _M> > : _is_monad_plus<_M> {};

template<typename S, typename _M>
struct MonadPlus<_StateT<S, _M> > : Monad<_StateT<S, _M> >, _MonadPlus<_StateT<S, _M> >
{
    using super = Monad<_StateT<S, _M> >;

    // mzero = StateT $ \ _ -> mzero
    template<typename A>
    static constexpr StateT<S, _M, A> mzero(){
        return _([](S const&){ return MonadPlus<_M>::mzero(); });
    }

    // StateT m `mplus` StateT n = StateT $ \ s -> m s `mplus` n s
    template<typename A>
    static constexpr StateT<S, _M, A> mplus(StateT<S, _M, A> const& m, StateT<S, _M, A> const& n){
        return _([m, n](S const& s){ return MonadPlus<_M>::mplus(m.run(s), n.run(s)); });
    }
};

// Alternative
template<typename S, typename _M>
struct _is_alternative<_StateT<S, _M> > : _is_monad_plus<_M> {};

template<typename S, typename _M>
struct Alternative<_StateT<S, _M> > : _Alternative<_StateT<S, _M> >
{
    // empty = StateT $ \ _ -> mzero
    template<typename A>
    static constexpr StateT<S, _M, A> empty(){
        return _([](S const&){ return MonadPlus<_M>::mzero(); });
    }

    // StateT m <|> StateT n = StateT $ \ s -> m s `mplus` n s
    template<typename A>
    static constexpr StateT<S, _M, A> alt_op(StateT<S, _M, A> const& m, StateT<S, _M, A> const& n){
        return _([m, n](S const& s){ return MonadPlus<_M>::mplus(m.run(s), n.run(s)); });
    }
};

// MonadTrans
template<typename S>
struct MonadTrans<__StateT<S> >
{
    // lift m = StateT $ \ s -> do
    //   a <- m
    //   return (a, s)
    template<typename M>
    static constexpr monad_type<M, StateT<S, base_class_t<M>, value_type_t<M> > > lift(M const& m){
        return _([m](S const& s){
            return _do(a, m,
                return Monad_t<M>::mreturn(pair_t<value_type_t<M>, S>(a, s)););
        });
    }
};

// MonadState
template<typename S, typename _M>
struct MonadState<S, _StateT<S, _M> > : _MonadState<S, _StateT<S, _M>, MonadState<S, _StateT<S, _M> > >
{
    using base_class = _StateT<S, _M>;
    using super = _MonadState<S, base_class, MonadState<S, base_class> >;

    template<typename A>
    static constexpr StateT<S, _M, A> state(function_t<pair_t<A, S>(S const&)> const& f){
        return base_class::state(f);
    }

    static constexpr StateT<S, _M, S>  get(){
        return base_class::get();
    }

    static constexpr StateT<S, _M, None> put(S const& s){
        return base_class::put(s);
    }
};

_FUNCPROG_END
