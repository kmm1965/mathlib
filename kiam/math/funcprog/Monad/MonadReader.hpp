#pragma once

#include "../funcprog_setup.h"

_FUNCPROG_BEGIN

/*
-- class MonadReader
--  asks for the internal (non-mutable) state.

-- | See examples in "Control.Monad.Reader".
-- Note, the partially applied function type @(->) r@ is a simple reader monad.
-- See the @instance@ declaration below.
class Monad m => MonadReader r m | m -> r where
*/
template<typename R, typename T>
struct MonadReader;

template<typename R, typename T>
using MonadReader_t = MonadReader<R, base_class_t<T> >;

template<typename R, typename _M, typename MR>
struct _MonadReader
{
    static_assert(_is_monad_v<_M>, "Should be a Monad");

    template<typename T>
    using type = typename _M::template type<T>;

    // -- | Retrieves the monad environment.
    // ask   :: m r
    // ask = reader id

    static constexpr type<R> ask(){
        return MR::reader(_(id<R>));
    }
    /*
    -- | Executes a computation in a modified environment.
    local :: (r -> r) -- ^ The function to modify the environment.
          -> m a      -- ^ @Reader@ to run in the modified environment.
          -> m a
    */
    /*
    -- | Retrieves a function of the current environment.
    reader :: (r -> a) -- ^ The selector function to apply to the environment.
           -> m a
    reader f = do
      r <- ask
      return (f r)
    */
    template<typename A, typename RArg>
    static constexpr std::enable_if_t<is_same_as_v<R, RArg>, type<A> >
    reader(function_t<A(RArg)> const& f){
        return _do(r, MR::ask(), return Monad<_M>::mreturn(f(r)););
    }
};

/*
-- | Retrieves a function of the current environment.
asks :: MonadReader r m
    => (r -> a) -- ^ The selector function to apply to the environment.
    -> m a
asks = reader
*/
template<typename R, typename _M, typename A, typename RArg>
std::enable_if_t<is_same_as_v<R, RArg>, typename _M::template type<A> >
asks(function_t<A(RArg)> const& f){
    return MonadReader<R, _M>::reader(f);
}

_FUNCPROG_END
