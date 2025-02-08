#pragma once

#include "Maybe_fwd.hpp"

_FUNCPROG_BEGIN

//-- |
//-- Wrapper for partial functions with 'MonadPlus' instance.
//--
//newtype Partial a b = Partial { getPartial :: a -> Maybe b }
template<typename R>
struct _Partial;

template<typename R, typename A>
struct Partial;

template<typename R, typename A>
constexpr Partial<R, A> Partial_(function_t<Maybe<A>(R const&)> const& f);

template<typename R, typename A>
constexpr function_t<Maybe<A>(R const&)> getPartial(Partial<R, A> const& partial);

// Functor
template<typename R>
struct _is_functor<_Partial<R> > : std::true_type {};

template<typename R, typename A>
struct is_functor<Partial<R, A> > : std::true_type {};

template<typename R>
struct Functor<_Partial<R> >;

// Applicative
template<typename R>
struct _is_applicative<_Partial<R> > : std::true_type {};

template<typename R, typename A>
struct is_applicative<Partial<R, A> > : std::true_type {};

template<typename R>
struct Applicative<_Partial<R> >;

// Monad
template<typename R>
struct _is_monad<_Partial<R> > : std::true_type {};

template<typename R, typename A>
struct is_monad<Partial<R, A> > : std::true_type {};

template<typename R>
struct Monad<_Partial<R> >;

// MonadPlus
template<typename R>
struct _is_monad_plus<_Partial<R> > : std::true_type {};

template<typename R>
struct MonadPlus<_Partial<R> >;

// Alternative
template<typename R>
struct _is_alternative<_Partial<R> > : std::true_type {};

template<typename R>
struct Alternative<_Partial<R> >;

// Monoid
template<typename R>
struct _is_monoid<_Partial<R> > : std::true_type {};

template<typename R>
struct Monoid<_Partial<R> >;

_FUNCPROG_END
