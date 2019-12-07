#pragma once

#include "int_constant.hpp"
#include "negate_expression.hpp"

_SYMDIFF1_BEGIN

template<class E1, char op, class E2>
struct additive_expression;

template<class E1, char op, class E2>
struct additive_expression_type
{
	typedef std::conditional_t<std::is_same<E2, int_constant<0> >::value, E1,
		std::conditional_t<std::is_same<E1, int_constant<0> >::value,
			std::conditional_t<op == '+', E2, negate_expression_t<E2> >,
			std::conditional_t<is_int_constant<E1>::value && is_int_constant<E2>::value,
				std::conditional_t<op == '+',
					int_constant<int_constant_value<E1>::value + int_constant_value<E2>::value>,
					int_constant<int_constant_value<E1>::value - int_constant_value<E2>::value>
				>,
				std::conditional_t<
					is_scalar<E1>::value && (std::is_same<E1, E2>::value || is_int_constant<E2>::value), E1,
					std::conditional_t<
						is_int_constant<E1>::value && is_scalar<E2>::value, E2,
						additive_expression<E1, op, E2>
					>
				>
			>
		>
	> type;
};

template<class E1, char op, class E2>
using additive_expression_t = typename additive_expression_type<E1, op, E2>::type;

template<char op, class E1, class E2>
constexpr typename std::enable_if<op == '+',
	additive_expression_t<typename E1::diff_type, op, typename E2::diff_type>
>::type
additive_expression_diff(const E1& e1, const E2& e2){
	return e1.diff() + e2.diff();
}

template<char op, class E1, class E2>
constexpr typename std::enable_if<op == '-',
	additive_expression_t<typename E1::diff_type, op, typename E2::diff_type>
>::type
additive_expression_diff(const E1& e1, const E2& e2){
	return e1.diff() - e2.diff();
}

template<class E1, char op, class E2>
struct additive_expression : expression<additive_expression<E1, op, E2> >
{
	typedef additive_expression_t<typename E1::diff_type, op, typename E2::diff_type> diff_type;
	
    constexpr additive_expression(const expression<E1> &e1, const expression<E2> &e2) : e1(e1()), e2(e2()){}

    constexpr const E1& expr1() const { return e1; }
    constexpr const E2& expr2() const { return e2; }

    constexpr diff_type diff() const {
		return additive_expression_diff<op>(e1, e2);
	}

	template<typename T>
    constexpr typename std::enable_if<op == '+', T>::type
	operator()(const T &x) const {
		return e1(x) + e2(x);
	}

	template<typename T>
    constexpr typename std::enable_if<op == '-', T>::type
	operator()(const T &x) const {
		return e1(x) - e2(x);
	}

    constexpr std::string to_string() const {
        return '(' + e1.to_string() + op + e2.to_string() + ')';
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
constexpr negate_expression_t<E>
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

_SYMDIFF1_END
