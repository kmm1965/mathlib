#pragma once

#include "../MonadError.hpp"

_FUNCPROG_BEGIN

template<typename _ME>
template<typename E, typename A>
constexpr typeof_t<_ME, A>
_MonadError<_ME>::liftEither(Either<E, A> const& x){
    return either(_(MonadError<_ME>::template throwError<A>), _(Monad<_ME>::template mreturn<A>), x);
}

template<typename M>
constexpr monad_type<M> liftEither(Either<MonadError_error_type<M>, value_type_t<M> > const& x){
    return MonadError_t<M>::template liftEither<MonadError_error_type<M> >(x);
}

template<typename M, typename A>
constexpr monad_type<M> throwError(A const& e){
    return MonadError_t<M>::template throwError<value_type_t<M> >(e);
}

FUNCTION_TEMPLATE(2) constexpr monad_type<T0> catchError(T0 const& x, function_t<T0(T1 const&)> f) {
    return MonadError_t<T0>::catchError(x, f);
}

_FUNCPROG_END
