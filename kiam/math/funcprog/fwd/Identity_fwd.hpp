#pragma once

#include "Functor_fwd.hpp"
#include "Applicative_fwd.hpp"
#include "Monad_fwd.hpp"
#include "Semigroup_fwd.hpp"
#include "Monoid_fwd.hpp"
#include "Foldable_fwd.hpp"
#include "Traversable_fwd.hpp"
#include "MonadZip_fwd.hpp"

_FUNCPROG_BEGIN

struct _Identity;

template<typename A>
struct Identity;

template<typename T>
constexpr T runIdentity(Identity<T> const&);

// Constructor
template<typename T>
constexpr Identity<T> Identity_(T const&);

template<typename T>
constexpr Identity<T> Identity_f(f0<T> const& fvalue);

// Functor
IMPLEMENT_FUNCTOR(_Identity, Identity);

template<>
struct Functor<_Identity>;

// Applicative
IMPLEMENT_APPLICATIVE(_Identity, Identity);

template<>
struct Applicative<_Identity>;

// Monad
IMPLEMENT_MONAD(_Identity, Identity);

template<>
struct Monad<_Identity>;

// Semigroup
IMPLEMENT_SEMIGROUP_COND(Identity);

template<>
struct Semigroup<_Identity>;

// Moniod
template<typename A> struct is_monoid<Identity<A> > : is_monoid<A> {};

template<>
struct Monoid<_Identity>;

// Foldable
IMPLEMENT_FOLDABLE(_Identity);

template<>
struct Foldable<_Identity>;

// Traversable
IMPLEMENT_TRAVERSABLE(_Identity);

template<>
struct Traversable<_Identity>;

// MonadZip
template<>
struct MonadZip<_Identity>;

template<typename T>
struct is_identity : std::false_type {};

template<typename A>
struct is_identity<Identity<A> > : std::true_type {};

_FUNCPROG_END

namespace std {

template<typename T>
ostream& operator<<(ostream& os, _FUNCPROG::Identity<T> const& v);

template<typename T>
wostream& operator<<(wostream& os, _FUNCPROG::Identity<T> const& v);

ostream& operator<<(ostream& os, _FUNCPROG::Identity<string> const& v);
wostream& operator<<(wostream& os, _FUNCPROG::Identity<wstring> const& v);
ostream& operator<<(ostream& os, _FUNCPROG::Identity<_FUNCPROG::f0<string> > const& v);
wostream& operator<<(wostream& os, _FUNCPROG::Identity<_FUNCPROG::f0<wstring> > const& v);

} // namespace std
