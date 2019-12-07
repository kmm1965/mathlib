#pragma once

#include "div_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct sqrt_expression : expression<sqrt_expression<E> >
{
	typedef typename mul_expression_type<
		typename div_expression_type<
			int_constant<1>,
			typename mul_expression_type<int_constant<2>, sqrt_expression<E> >::type
		>::type,
		typename E::diff_type
	>::type diff_type;

    constexpr sqrt_expression(const expression<E> &e) : e(e()) {}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
		return int_constant<1>() / (int_constant<2>() * sqrt(e)) * e.diff();
	}

	template<typename T>
    constexpr T operator()(const T &x) const {
		return std::sqrt(e(x));
	}

    constexpr std::string to_string() const {
        return "sqrt(" + e.to_string() + ')';
    }

private:
	const E e;
};

template<class E>
constexpr sqrt_expression<E> sqrt(const expression<E>& e) {
	return sqrt_expression<E>(e);
}

_SYMDIFF1_END
