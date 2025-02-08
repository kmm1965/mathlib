#pragma once

#include "Functor_fwd.hpp"
#include "Applicative_fwd.hpp"
#include "MonadError_fwd.hpp"
#include "Monad_fwd.hpp"
#include "Alternative_fwd.hpp"
#include "MonadPlus_fwd.hpp"
#include "Semigroup_fwd.hpp"
#include "Monoid_fwd.hpp"
#include "Foldable_fwd.hpp"
#include "Traversable_fwd.hpp"
#include "MonadZip_fwd.hpp"
#include "../MonadFail.hpp"

_FUNCPROG_BEGIN

class maybe_error;
class maybe_nothing_error;

struct _Maybe;

template<typename A>
struct Maybe;

// Constructors
template<typename T> constexpr Maybe<fdecay<T> > Just(T const&);
template<typename T> constexpr Maybe<T> Nothing();
template<typename T> constexpr f0<Maybe<T> > _Nothing();

DECLARE_FUNCTION_3(2, T1, maybe, T1 const&, function_t<T1(T0)> const&, Maybe<fdecay<T0> > const&);
template<typename T> constexpr bool isJust(Maybe<T> const&);
template<typename T> constexpr bool isNothing(Maybe<T> const&);
template<typename T> constexpr T fromJust(Maybe<T> const&);
DECLARE_FUNCTION_2(1, T0, fromMaybe, T0 const&, Maybe<T0> const&);

template<typename A> constexpr A fromMaybe(f0<A> const&, Maybe<A> const&);
template<typename A> constexpr function_t<A(Maybe<A> const&)> _fromMaybe(f0<A> const& f);

// Functor
IMPLEMENT_FUNCTOR(_Maybe, Maybe);

template<>
struct Functor<_Maybe>;

// Applicative
IMPLEMENT_APPLICATIVE(_Maybe, Maybe);

template<>
struct Applicative<_Maybe>;

// MonadFail
template<>
struct MonadFail<_Maybe>;

// Monad
IMPLEMENT_MONAD(_Maybe, Maybe);

template<>
struct Monad<_Maybe>;

// Alternative
IMPLEMENT_ALTERNATIVE(_Maybe);

template<>
struct Alternative<_Maybe>;

// MonadPlus
IMPLEMENT_MONADPLUS(_Maybe);

template<>
struct MonadPlus<_Maybe>;

// Semigroup
IMPLEMENT_SEMIGROUP_COND(Maybe);

template<>
struct Semigroup<_Maybe>;

// Monoid
template<typename A> struct is_monoid<Maybe<A> > : is_semigroup<A> {};

template<>
struct Monoid<_Maybe>;

// Foldable
IMPLEMENT_FOLDABLE(_Maybe);

template<>
struct Foldable<_Maybe>;

// Traversable
IMPLEMENT_TRAVERSABLE(_Maybe);

template<>
struct Traversable<_Maybe>;

// MonadError
template<>
struct MonadError<_Maybe>;

// MonadZip
template<>
struct MonadZip<_Maybe>;

template<typename T>
struct is_maybe : std::false_type {};

template<typename A>
struct is_maybe<Maybe<A> > : std::true_type {};

template<typename T>
bool constexpr is_maybe_v = is_maybe<T>::value;

_FUNCPROG_END

namespace std {

template<typename T>
ostream& operator<<(ostream& os, _FUNCPROG::Maybe<T> const& mv);

template<typename T>
wostream& operator<<(wostream& os, _FUNCPROG::Maybe<T> const& mv);

ostream& operator<<(ostream& os, _FUNCPROG::Maybe<string> const& mv);
wostream& operator<<(wostream& os, _FUNCPROG::Maybe<wstring> const& mv);
ostream& operator<<(ostream& os, _FUNCPROG::Maybe<_FUNCPROG::f0<string> > const& mv);
wostream& operator<<(wostream& os, _FUNCPROG::Maybe<_FUNCPROG::f0<wstring> > const& mv);

template<typename A>
std::ostream& operator<<(std::ostream& os, _FUNCPROG::EmptyData<A> const&);

template<typename A>
std::wostream& operator<<(std::wostream& os, _FUNCPROG::EmptyData<A> const&);

} // namespace std
