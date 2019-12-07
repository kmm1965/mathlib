#pragma once

#include "func.hpp"

_FUNCPROG_BEGIN

template<typename A, typename B>
using pair_t = std::pair<A, B>;

#define PAIR_T_(A, B) BOOST_IDENTITY_TYPE((pair_t<A, B>))
#define PAIR_T(A, B) typename PAIR_T_(A, B)

template<typename T>
struct is_pair : std::false_type {};

template<typename A, typename B>
struct is_pair<pair_t<A, B> > : std::true_type {};

template<typename T>
struct fst_type;

template<typename A, typename B>
struct fst_type<pair_t<A, B> > {
	using type = A;
};

template<typename T>
using fst_type_t = typename fst_type<T>::type;

template<typename T>
struct snd_type;

template<typename A, typename B>
struct snd_type<pair_t<A, B> > {
	using type = B;
};

template<typename T>
using snd_type_t = typename snd_type<T>::type;

template<typename A, typename B, typename C>
using tuple3_t = std::tuple<A, B, C>;

#define TUPLE3_(T0, T1, T2) BOOST_IDENTITY_TYPE((tuple3_t<T0, T1, T2>))
#define TUPLE3(T0, T1, T2) typename TUPLE3_(T0, T1, T2)

template<typename T>
struct is_tuple3 : std::false_type {};

template<typename A, typename B, typename C>
struct is_tuple3<tuple3_t<A, B, C> > : std::true_type {};

template<typename T>
T id(T const& value) {
	return value;
}

DEFINE_FUNCTION_2(2, T1, const_, T1 const&, value, T0 const&, __unused__, return value;)
DEFINE_FUNCTION_3(3, T0, flip, function_t<T0(T1, T2)> const&, f, fdecay<T2> const&, y, fdecay<T1> const&, x, return f(x, y);)

#define DEFINE_LOG_OPERATION(name, op) \
	DEFINE_FUNCTION_2(1, bool, name, T0 const&, l, T0 const&, r, return l op r;)

DEFINE_LOG_OPERATION(eq, ==)
DEFINE_LOG_OPERATION(neq, !=)
DEFINE_LOG_OPERATION(lt, <)
DEFINE_LOG_OPERATION(le, <=)
DEFINE_LOG_OPERATION(gt, >)
DEFINE_LOG_OPERATION(ge, >=)

enum Ordering { EQ, LT, GT };

DEFINE_FUNCTION_2(1, Ordering, compare, T0 const&, v1, T0 const&, v2,
    return v1 == v2 ? EQ : v1 < v2 ? LT : GT;)

template<typename... Args>
remove_f0_t<function_t<bool(Args...)> > not_(function_t<bool(Args...)> const& f) {
    return invoke_f0(_([f](Args... args) { return !f(args...); }));
}

template<class C1, class C2>
struct is_same_class : std::is_same<C1, C2> {};

_FUNCPROG_END

namespace std {

	template<typename T>
	ostream& operator<<(ostream& os, _FUNCPROG::fdata<T> const& v) {
		return os << v();
	}

	template<typename T>
	wostream& operator<<(wostream& os, _FUNCPROG::fdata<T> const& v) {
		return os << v();
	}

}
