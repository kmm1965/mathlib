#pragma once

#include "Functor_fwd.hpp"
#include "Applicative_fwd.hpp"
#include "Monad_fwd.hpp"
//#include "Alternative_fwd.hpp"
//#include "MonadPlus_fwd.hpp"
#include "Semigroup_fwd.hpp"
#include "Monoid_fwd.hpp"
#include "Foldable_fwd.hpp"
#include "Traversable_fwd.hpp"
#include "MonadZip_fwd.hpp"
#include "../MonadFail.hpp"

_FUNCPROG2_BEGIN

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
//IMPLEMENT_ALTERNATIVE(_List);

//template<>
//struct Alternative<_List>;

// MonadPlus
//IMPLEMENT_MONADPLUS(_List);

//template<>
//struct MonadPlus<_List>;

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

template<typename A, typename B, typename FuncImpl1, typename FuncImpl2>
constexpr List<A> build(function2<B(function2<B(A const&, B const&), FuncImpl1> const&, B const&), FuncImpl2> const& g);

template<typename A, typename B, typename FuncImpl1, typename FuncImpl2>
constexpr List<A> augment(function2<B(function2<B(A const&, B const&), FuncImpl1> const&, B const&), FuncImpl2> const& g, List<A> const& xs);

_FUNCPROG2_END

namespace std {

template<typename A>
ostream& operator<<(ostream& os, _FUNCPROG2::List<A> const& l);

template<typename A>
wostream& operator<<(wostream& os, _FUNCPROG2::List<A> const& l);

template<typename A, typename FuncImpl>
ostream& operator<<(ostream& os, _FUNCPROG2::List<_FUNCPROG2::f0<A, FuncImpl> > const& lf);

template<typename A, typename FuncImpl>
wostream& operator<<(wostream& os, _FUNCPROG2::List<_FUNCPROG2::f0<A, FuncImpl> > const& lf);

inline ostream& operator<<(ostream& os, _FUNCPROG2::String const& s);
inline wostream& operator<<(wostream& os, _FUNCPROG2::wString const& s);
inline ostream& operator<<(ostream& os, _FUNCPROG2::List<string> const& ls);
inline wostream& operator<<(wostream& os, _FUNCPROG2::List<wstring> const& ls);

template<typename FuncImpl>
inline ostream& operator<<(ostream& os, _FUNCPROG2::List<_FUNCPROG2::f0<string, FuncImpl> > const& lf);

template<typename FuncImpl>
inline wostream& operator<<(wostream& os, _FUNCPROG2::List<_FUNCPROG2::f0<wstring, FuncImpl> > const& lf);

} // namespace std

#include "ListApi_fwd.hpp"
