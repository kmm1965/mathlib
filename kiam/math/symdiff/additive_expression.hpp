#pragma once

#include "int_constant.hpp"
#include "negate_expression.hpp"

_SYMDIFF_BEGIN

template<class E1, char op, class E2>
struct additive_expression;

template<class E1, char op, class E2>
struct additive_expression_type
{
	typedef typename std::conditional<std::is_same<E2, int_constant<0> >::value, E1,
		typename std::conditional<std::is_same<E1, int_constant<0> >::value,
			typename std::conditional<op == '+', E2, typename negate_expression_type<E2>::type>::type,
			typename std::conditional<is_int_constant<E1>::value && is_int_constant<E2>::value,
				typename std::conditional<op == '+',
					int_constant<int_constant_value<E1>::value + int_constant_value<E2>::value>,
					int_constant<int_constant_value<E1>::value - int_constant_value<E2>::value>
				>::type,
				typename std::conditional<
					is_scalar<E1>::value && (std::is_same<E1, E2>::value || is_int_constant<E2>::value), E1,
					typename std::conditional<is_int_constant<E1>::value && is_scalar<E2>::value, E2,
						additive_expression<E1, op, E2>
					>::type
				>::type
			>::type
		>::type
	>::type type;
};

template<class E1, char op, class E2>
struct additive_expression : expression<additive_expression<E1, op, E2> >
{
	template<unsigned M>
	struct diff_type
	{
		typedef typename additive_expression_type<
			typename E1::template diff_type<M>::type,
			op,
			typename E2::template diff_type<M>::type
		>::type type;
	};
	
    constexpr additive_expression(const expression<E1> &e1, const expression<E2> &e2) : e1(e1()), e2(e2()){}

    constexpr const E1& expr1() const { return e1; }
    constexpr const E2& expr2() const { return e2; }

	template<unsigned M>
    constexpr typename std::enable_if<op == '+', typename diff_type<M>::type>::type
    diff() const {
		return e1.diff<M>() + e2.diff<M>();
	}

	template<unsigned M>
    constexpr typename std::enable_if<op == '-', typename diff_type<M>::type>::type
	diff() const {
		return e1.diff<M>() - e2.diff<M>();
	}

	template<typename T, size_t _Size>
    constexpr typename std::enable_if<op == '+', T>::type
	operator()(const std::array<T, _Size> &vars) const {
		return e1(vars) + e2(vars);
	}

	template<typename T, size_t _Size>
    constexpr typename std::enable_if<op == '-', T>::type
	operator()(const std::array<T, _Size> &vars) const {
		return e1(vars) - e2(vars);
	}

private:
	const E1 e1;
	const E2 e2;
};

template<class E1, class E2>
constexpr additive_expression<E1, '+', E2>
operator+(const expression<E1> &e1, const expression<E2> &e2){
	return additive_expression<E1, '+', E2>(e1, e2);
}

template<class E1, class E2>
constexpr additive_expression<E1, '-', E2>
operator-(const expression<E1> &e1, const expression<E2> &e2){
	return additive_expression<E1, '-', E2>(e1, e2);
}

template<int N1, int N2>
constexpr int_constant<N1 + N2>
operator+(const int_constant<N1>&, const int_constant<N2>&){
	return int_constant<N1 + N2>();
}

template<int N1, int N2>
constexpr int_constant<N1 - N2>
operator-(const int_constant<N1>&, const int_constant<N2>&){
	return int_constant<N1 - N2>();
}

template<class E>
constexpr const E& operator+(const expression<E> &e, const int_constant<0>&){
	return e();
}

template<class E>
constexpr const E& operator-(const expression<E> &e, const int_constant<0>&){
	return e();
}

template<class E>
constexpr const E& operator+(const int_constant<0>&, const expression<E> &e){
	return e();
}

template<class E>
constexpr typename negate_expression_type<E>::type
operator-(const int_constant<0>&, const expression<E> &e){
	return -e;
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
	additive_expression<E, '+', scalar<T> >
>::type operator+(const expression<E> &e, const T &val){
	return e + scalar<T>(val);
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
	additive_expression<scalar<T>, '+', E>
>::type operator+(const T &val, const expression<E> &e){
	return scalar<T>(val) + e;
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
	additive_expression<E, '-', scalar<T> >
>::type operator-(const expression<E> &e, const T &val){
	return e - scalar<T>(val);
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
	additive_expression<scalar<T>, '-', E>
>::type operator-(const T &val, const expression<E> &e){
	return scalar<T>(val) - e;
}

template<typename VT>
constexpr scalar<VT>
operator+(const scalar<VT> &e1, const scalar<VT> &e2){
	return scalar<VT>(e1.value + e2.value);
}

template<typename VT>
constexpr scalar<VT>
operator-(const scalar<VT> &e1, const scalar<VT> &e2){
	return scalar<VT>(e1.value - e2.value);
}

template<typename VT, int N>
constexpr scalar<VT>
operator+(const scalar<VT> &e1, const int_constant<N>&){
	return scalar<VT>(e1.value + N);
}

template<typename VT, int N>
constexpr scalar<VT>
operator-(const scalar<VT> &e1, const int_constant<N>&){
	return scalar<VT>(e1.value - N);
}

template<int N, typename VT>
constexpr scalar<VT>
operator+(const int_constant<N>&, const scalar<VT> &e2){
	return scalar<VT>(N + e2.value);
}

template<int N, typename VT>
constexpr scalar<VT>
operator-(const int_constant<N>&, const scalar<VT> &e2){
	return scalar<VT>(N - e2.value);
}

_SYMDIFF_END
