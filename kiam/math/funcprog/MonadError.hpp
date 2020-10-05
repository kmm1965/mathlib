#pragma once

#include "Either_fwd.hpp"

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

// requires throwError, catchError
template<typename ME>
struct MonadError;

template<typename T>
using MonadError_t = MonadError<base_class_t<T> >;

#define DECLARE_MONADERROR_CLASS(ME) \
    /* throwError :: e -> m a */ \
    template<typename A> \
    static ME<A> throwError(error_type<A> const&); \
    /* catchError :: m a -> (e -> m a) -> m a */ \
    template<typename A> \
    static ME<A> catchError(ME<A> const& x, function_t<ME<A>(error_type<A> const&)> const& f);

template<typename _ME>
struct MonadError_base
{
    static_assert(is_monad<_ME>::value, "Should be a monad");
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
    static typeof_t<_ME, A> liftEither(Either<E, A> const& x) {
        return either(_(MonadError<_ME>::template throwError<A>), _(Monad<_ME>::template mreturn<A>), x);
    }
};

template<typename M>
using MonadError_error_type = typename MonadError_t<M>::template error_type<value_type_t<M> >;

template<typename M>
monad_type<M> liftEither(Either<MonadError_error_type<M>, value_type_t<M> > const& x) {
    return MonadError_t<M>::template liftEither<MonadError_error_type<M> >(x);
}

template<typename M, typename A>
monad_type<M> throwError(A const& e){
    return MonadError_t<M>::template throwError<value_type_t<M> >(e);
}

DEFINE_FUNCTION_2(2, monad_type<T0>, catchError, T0 const&, x, function_t<T0(T1 const&)>, f,
    return MonadError_t<T0>::catchError(x, f);)

_FUNCPROG_END
