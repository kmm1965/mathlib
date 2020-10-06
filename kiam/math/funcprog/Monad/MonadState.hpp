#pragma once

#include "../funcprog_setup.h"

_FUNCPROG_BEGIN

/*
-- | Minimal definition is either both of @get@ and @put@ or just @state@
class Monad m => MonadState s m | m -> s where
    -- | Return the state from the internals of the monad.
    get :: m s
    get = state (\s -> (s, s))

    -- | Replace the state inside the monad.
    put :: s -> m ()
    put s = state (\_ -> ((), s))

    -- | Embed a simple state action into the monad.
    state :: (s -> (a, s)) -> m a
    state f = do
      s <- get
      let ~(a, s') = f s
      put s'
      return a
*/
template<typename S, typename T>
struct MonadState;

template<typename S, typename T>
using MonadState_t = MonadState<S, base_class_t<T> >;

template<typename S, typename _M, typename MS>
struct _MonadState
{
    static_assert(_is_monad<_M>::value, "Should be a Monad");

    template<typename T>
    using type = typename _M::template type<T>;

    /*
    -- | Embed a simple state action into the monad.
    state :: (s -> (a, s)) -> m a
    state f = do
      s <- get
      let ~(a, s') = f s
      put s'
      return a
    */
    template<typename A>
    static type<A> state(function_t<pair_t<A, S>(S const&)> const& f)
    {
        using pair_type = pair_t<A, S>;
        return _do(s, MS::get(),
            const pair_type p = f(s);
            return MS::put(snd(p)) >> Monad<_M>::mreturn(fst(p));
        );
    }

    /*
    -- | Return the state from the internals of the monad.
    get :: m s
    get = state (\s -> (s, s))
    */
    static type<S> get() {
        return state<S>(_([](S const& s) { return pair_t<S, S>(s, s); }));
    }

    /*
    -- | Replace the state inside the monad.
    put :: s -> m ()
    put s = state (\_ -> ((), s))
    */
    static type<None> put(S const& s) {
        return state<None>(_([s](S const&) { return pair_t<None, S>(None(), s); }));
    }

    /*
    -- | Monadic state transformer.
    --
    --      Maps an old state to a new state inside a state monad.
    --      The old state is thrown away.
    --
    -- >      Main> :t modify ((+1) :: Int -> Int)
    -- >      modify (...) :: (MonadState Int a) => a ()
    --
    --    This says that @modify (+1)@ acts over any
    --    Monad that is a member of the @MonadState@ class,
    --    with an @Int@ state.
    modify :: MonadState s m => (s -> s) -> m ()
    modify f = state (\s -> ((), f s))
    */
    static type<None> modify(function_t<S(S const&)> const& f) {
        return state<None>(_([f](S const& s) { return pair_t<None, S>(None(), f(s)); }));
    }

    /*
    -- | Gets specific component of the state, using a projection function
    -- supplied.
    gets :: MonadState s m => (s -> a) -> m a
    gets f = do
        s <- get
        return (f s)
    */
    template<typename A>
    static type<A> gets(function_t<A(S const&)> const& f) {
        return _do(s, MS::get(), return Monad<_M>::mreturn(f(s)););
    }
};

_FUNCPROG_END
