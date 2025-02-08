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
template<typename _ME>
struct MonadError;

template<typename T>
using MonadError_t = MonadError<base_class_t<T> >;

template<typename _ME>
struct _MonadError;

template<typename M>
using MonadError_error_type = typename MonadError_t<M>::template error_type<value_type_t<M> >;

template<typename M>
constexpr monad_type<M> liftEither(Either<MonadError_error_type<M>, value_type_t<M> > const& x);

template<typename M, typename A>
constexpr monad_type<M> throwError(A const& e);

DECLARE_FUNCTION_2(2, monad_type<T0>, catchError, T0 const&, function_t<T0(T1 const&)>);

_FUNCPROG_END
