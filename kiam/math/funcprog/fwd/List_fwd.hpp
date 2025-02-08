#pragma once

#include "Functor_fwd.hpp"
#include "Applicative_fwd.hpp"
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

class list_error;
class empty_list_error;

struct _List;

template<typename T>
struct List;

using String = List<char>;
using wString = List<wchar_t>;

// Functor
IMPLEMENT_FUNCTOR(_List, List);

template<>
struct Functor<_List>;

// Applicative
IMPLEMENT_APPLICATIVE(_List, List);

template<>
struct Applicative<_List>;

// MonadFail
template<>
struct MonadFail<_List>;

// Monad
IMPLEMENT_MONAD(_List, List);

template<>
struct Monad<_List>;

// Alternative
IMPLEMENT_ALTERNATIVE(_List);

template<>
struct Alternative<_List>;

// MonadPlus
IMPLEMENT_MONADPLUS(_List);

template<>
struct MonadPlus<_List>;

// Semigroup
IMPLEMENT_SEMIGROUP(_List);

template<>
struct Semigroup<_List>;

// Monoid
IMPLEMENT_MONOID(_List);

template<>
struct Monoid<_List>;

// Foldable
IMPLEMENT_FOLDABLE(_List);

template<>
struct Foldable<_List>;

// Traversable
IMPLEMENT_TRAVERSABLE(_List);

template<>
struct Traversable<_List>;

// MonadZip
template<>
struct MonadZip<_List>;

// List
template<typename T>
struct is_list : std::false_type {};

template<typename A>
struct is_list<List<A> > : std::true_type {};

template<typename T>
bool constexpr is_list_v = is_list<T>::value;

template<typename L>
struct list_value_type {
    typedef void type;
};

template<typename A>
struct list_value_type<List<A> > {
    typedef A type;
};

template<typename T>
constexpr List<T> operator>>(T const& value, List<T> const& l);

List<char> operator>>(char value, List<char> const& l);
List<wchar_t> operator>>(wchar_t value, List<wchar_t> const& l);

template<typename T>
constexpr List<T> operator<<(List<T> const& l, T const& value);

template<typename A, typename B>
constexpr List<A> build(function_t<B(function_t<B(A const&, B const&)> const&, B const&)> const& g);

template<typename A, typename B>
constexpr List<A> augment(function_t<B(function_t<B(A const&, B const&)> const&, B const&)> const& g, List<A> const& xs);

_FUNCPROG_END

namespace std {

template<typename A>
ostream& operator<<(ostream& os, _FUNCPROG::List<A> const& l);

template<typename A>
wostream& operator<<(wostream& os, _FUNCPROG::List<A> const& l);

template<typename A>
ostream& operator<<(ostream& os, _FUNCPROG::List<_FUNCPROG::f0<A> > const& lf);

template<typename A>
wostream& operator<<(wostream& os, _FUNCPROG::List<_FUNCPROG::f0<A> > const& lf);

inline ostream& operator<<(ostream& os, _FUNCPROG::String const& s);
inline wostream& operator<<(wostream& os, _FUNCPROG::wString const& s);
inline ostream& operator<<(ostream& os, _FUNCPROG::List<string> const& ls);
inline wostream& operator<<(wostream& os, _FUNCPROG::List<wstring> const& ls);
inline ostream& operator<<(ostream& os, _FUNCPROG::List<_FUNCPROG::f0<string> > const& lf);
inline wostream& operator<<(wostream& os, _FUNCPROG::List<_FUNCPROG::f0<wstring> > const& lf);

} // namespace std

#include "ListApi_fwd.hpp"
