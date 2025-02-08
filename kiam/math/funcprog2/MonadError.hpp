#pragma once

#include "fwd/MonadError_fwd.hpp"

_FUNCPROG_BEGIN

/*
    -- | Is used within a monadic computation to begin exception processing.
    throwError :: e -> m a

    {- |
    A handler function to handle previous errors and return to normal execution.
    A common idiom is:

    > do { action1; action2; action3 } `catchError` handler

    where the @action@ functions can call 'throwError'.
    Note that @handler@ and the do-block must have the same return type.
    -}
    catchError :: m a -> (e -> m a) -> m a
*/

template<typename _ME>
struct _MonadError
{
    static_assert(_is_monad_v<_ME>, "Should be a monad");
/*
Lifts an @'Either' e@ into any @'MonadError' e@.

> do { val <- liftEither =<< action1; action2 }

where @action1@ returns an 'Either' to represent errors.

@since 2.2.2
-}
liftEither :: MonadError e m => Either e a -> m a
liftEither = either throwError return
*/
    template<typename E, typename A>
    static constexpr typeof_t<_ME, A> liftEither(Either<E, A> const& x);
};

#define DECLARE_MONADERROR_CLASS(ME) \
    /* throwError :: e -> m a */ \
    template<typename A> \
    static constexpr ME<A> throwError(error_type<A> const&); \
    /* catchError :: m a -> (e -> m a) -> m a */ \
    template<typename A> \
    static constexpr ME<A> catchError(ME<A> const& x, function_t<ME<A>(error_type<A> const&)> const& f);

_FUNCPROG_END
