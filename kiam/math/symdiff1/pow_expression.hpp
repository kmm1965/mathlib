#pragma once

#include "int_constant.hpp"
#include "mul_expression.hpp"

_SYMDIFF1_BEGIN

template<class E, int N>
struct pow_expression;

template<int N, class E>
constexpr typename std::enable_if<N == 0, int_constant<1> >::type
pow(const expression<E>& e){
	return int_constant<0>();
}

template<int N, class E>
constexpr typename std::enable_if<N == 1, const E& >::type
pow(const expression<E>& e){
	return e();
}

template<int N, class E>
constexpr typename std::enable_if<N != 0 && N != 1, pow_expression<E, N> >::type
pow(const expression<E>& e);

template<int N, class E, int M>
constexpr typename std::enable_if<N != 0 && N != 1, pow_expression<E, N * M> >::type
pow(const pow_expression<E, M>& e);

template<typename E>
struct is_pow_expression : std::false_type {};

template<typename E, int N>
struct is_pow_expression<pow_expression<E, N> > : std::true_type {};

template<typename E>
struct pow_expression_expr_type {
	typedef void type;
};

template<typename E, int N>
struct pow_expression_expr_type<pow_expression<E, N> > {
	typedef E type;
};

template<typename E>
struct pow_expression_index : std::integral_constant<int, 0> {};

template<typename E, int N>
struct pow_expression_index<pow_expression<E, N> > : std::integral_constant<int, N> {};

template<class E, int N>
struct pow_expression_type
{
	typedef typename std::conditional<
		N == 0, int_constant<1>,
		typename std::conditional<
			N == 1, E,
			typename std::conditional<
				is_pow_expression<E>::value,
				pow_expression<typename pow_expression_expr_type<E>::type, N * pow_expression_index<E>::value>,
				pow_expression<E, N>
			>::type
		>::type
	>::type type;
};

template<class E, int N>
using pow_expression_t = typename pow_expression_type<E, N>::type;

template<class E, int N>
struct pow_expression : expression<pow_expression<E, N> >
{
	typedef pow_expression type;

	typedef typename mul_expression_type<
		typename mul_expression_type<
			int_constant<N>,
			typename pow_expression_type<E, N - 1>::type
		>::type,
		typename E::diff_type
	>::type diff_type;

    constexpr pow_expression(const expression<E>& e) : e(e()){}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
		return int_constant<N>() * kiam::math::symdiff1::pow<N - 1>(e) * e.diff();
	}

	template<typename T>
    constexpr T operator()(const T &x) const {
		return _KIAM_MATH::math_pow<N>(e(x));
	}

    constexpr std::string to_string() const
    {
        std::ostringstream ss;
        ss << N;
        return e.to_string() + '^' + ss.str();
    }

private:
	const E e;
};

template<int N, class E>
constexpr typename std::enable_if<N != 0 && N != 1, pow_expression<E, N> >::type
pow(const expression<E>& e){
	return pow_expression<E, N>(e);
}

template<int N, class E, int M>
constexpr typename std::enable_if<N != 0 && N != 1, pow_expression<E, N * M> >::type
pow(const pow_expression<E, M>& e) {
	return pow_expression<E, N * M>(e.expr());
}

_SYMDIFF1_END
