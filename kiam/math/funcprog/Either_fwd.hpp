#pragma once

#include "funcprog_setup.h"

_FUNCPROG_BEGIN

template<typename A>
struct _Either;

template<typename A, typename B>
struct Either;

#define EITHER_(A, B) BOOST_IDENTITY_TYPE((Either<A, B>))
#define EITHER(A, B) typename EITHER_(A, B)

template<typename A>
struct _Left;

template<typename B>
struct _Right;

enum { Left_, Right_ };

// Constructors
template<typename A>
Either<A, void> Left(A const& value);

template<typename A>
Either<A, void> Left(f0<A> const& value);

template<typename B>
Either<void, B> Right(B const& value);

template<typename B>
Either<void, B> Right(f0<B> const& value);

// Either
// either                  :: (a -> c) -> (b -> c) -> Either a b -> c
// either f _ (Left x)     =  f x
// either _ g (Right y)    =  g y
DECLARE_FUNCTION_3_ARGS(3, remove_f0_t<function_t<T2(Args...)> >, either, function_t<T2(T0, Args...)> const&, function_t<T2(T1, Args...)> const&,
    EITHER(fdecay<T0>, fdecay<T1>) const&);

_FUNCPROG_END

namespace std {

template<typename A, typename B>
ostream& operator<<(ostream& os, _FUNCPROG::Either<A, B> const& v);

template<typename A, typename B>
wostream& operator<<(wostream& os, _FUNCPROG::Either<A, B> const& v);

} // namespace std
