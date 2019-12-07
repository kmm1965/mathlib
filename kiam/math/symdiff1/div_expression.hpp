#pragma once

#include "pow_expression.hpp"

_SYMDIFF1_BEGIN

template<class E1, class E2>
struct div_expression;

template<class E1, class E2>
struct div_expression_type
{
	typedef std::conditional_t<
		std::is_same<E1, int_constant<0> >::value, int_constant<0>,
		std::conditional_t<std::is_same<E2, int_constant<0> >::value, void,
			std::conditional_t<std::is_same<E2, int_constant<1> >::value, E1,
				std::conditional_t<
					is_scalar<E1>::value && (std::is_same<E1, E2>::value || is_int_constant<E2>::value), E1,
					std::conditional_t<
						is_int_constant<E1>::value && is_scalar<E2>::value, E2,
						div_expression<E1, E2>
					>
				>
			>
		>
	> type;
};

template<class E1, class E2>
using div_expression_t = typename div_expression_type<E1, E2>::type;

template<class E1, class E2>
typename std::enable_if<is_int_constant<E2>::value, div_expression_t<typename E1::diff_type, E2> >::type
div_diff(const E1 &e1, const E2 &e2){
	return e1.diff() / e2;
}

template<class E1, class E2>
typename std::enable_if<!is_int_constant<E2>::value,
	div_expression_t<
		additive_expression_t<
			mul_expression_t<typename E1::diff_type, E2>,
			'-',
			mul_expression_t<E1, typename E2::diff_type>
		>,
		pow_expression_t<E2, 2>
	>
>::type
div_diff(const E1 &e1, const E2 &e2){
	return (e1.diff() * e2 - e1 * e2.diff()) / kiam::math::symdiff1::pow<2>(e2);
}

template<class E1, class E2>
struct div_expression : expression<div_expression<E1, E2> >
{
	typedef std::conditional_t<is_int_constant<E2>::value,
		div_expression_t<typename E1::diff_type, E2>,
		div_expression_t<
			additive_expression_t<
				mul_expression_t<typename E1::diff_type, E2>,
				'-',
				mul_expression_t<E1, typename E2::diff_type>
			>,
			pow_expression_t<E2, 2>
		>
	> diff_type;

    constexpr div_expression(const expression<E1> &e1, const expression<E2> &e2) : e1(e1()), e2(e2()){}

    constexpr const E1& expr1() const { return e1; }
    constexpr const E2& expr2() const { return e2; }

    constexpr diff_type diff() const {
		return div_diff(e1, e2);
	}

    template<typename T>
    constexpr T operator()(const T &x) const {
		return e1(x) / e2(x);
	}

    constexpr std::string to_string() const {
        return '(' + e1.to_string() + '/' + e2.to_string() + ')';
    }

private:
	const E1 e1;
	const E2 e2;
};

template<class E1, class E2>
constexpr div_expression<E1, E2>
operator/(const expression<E1>& e1, const expression<E2>& e2){
	return div_expression<E1, E2>(e1, e2);
}

template<class E>
constexpr void operator/(const expression<E>&, const int_constant<0>&){
	// static_assert(false, "Zero division in operator/");
}

template<class E>
constexpr const int_constant<0>
operator/(const int_constant<0>&, const expression<E>&){
	return int_constant<0>();
}

template<class E>
constexpr const E& operator/(const expression<E> &e, const int_constant<1>&){
	return e();
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
	div_expression<E, scalar<T> >
>::type operator/(const expression<E> &e, const T &val){
	return e / scalar<T>(val);
}

template<class E, typename T>
constexpr typename std::enable_if<std::is_arithmetic<T>::value,
	div_expression<scalar<T>, E>
>::type operator/(const T &val, const expression<E> &e){
	return scalar<T>(val) / e;
}

template<typename VT>
constexpr scalar<VT> operator/(const scalar<VT> &e1, const scalar<VT> &e2){
	return scalar<VT>(e1.value / e2.value);
}

template<typename VT, int N>
constexpr typename std::enable_if<N != 0, scalar<VT> >::type
operator/(const scalar<VT> &e1, const int_constant<N>&){
	return scalar<VT>(e1.value / N);
}

template<int N, typename VT>
constexpr typename std::enable_if<N != 0, scalar<VT> >::type
operator/(const int_constant<N>&, const scalar<VT> &e2){
	return scalar<VT>(N / e2.value);
}

_SYMDIFF1_END
