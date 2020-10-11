#pragma once

#include "funcprog_setup.h"

_FUNCPROG_BEGIN

class maybe_error;
class maybe_nothing_error;

struct _Maybe;

template<typename T>
struct Maybe;

// Constructors
template<typename T> constexpr Maybe<fdecay<T> > Just(T const&);
template<typename T> constexpr Maybe<T> Nothing();
template<typename T> constexpr f0<Maybe<T> > _Nothing();

DECLARE_FUNCTION_3(2, Maybe<T1>, maybe, T1 const&, function_t<T1(T0)> const&, Maybe<fdecay<T0> > const&);
template<typename T> constexpr bool isJust(Maybe<T> const&);
template<typename T> constexpr bool isNothing(Maybe<T> const&);
template<typename T> constexpr T fromJust(Maybe<T> const&);
DECLARE_FUNCTION_2(1, T0, fromMaybe, T0 const&, Maybe<T0> const&);

template<typename T0> constexpr T0 fromMaybe(f0<T0> const&, Maybe<T0> const&);

template<typename A>
struct EmptyData {};

using None = EmptyData<void>;

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
