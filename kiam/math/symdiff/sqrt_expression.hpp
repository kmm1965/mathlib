#pragma once

#include "div_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct sqrt_expression : expression<sqrt_expression<E> >
{
	template<unsigned M>
	struct diff_type
	{
		typedef typename mul_expression_type<
			typename div_expression_type<
				int_constant<1>,
				typename mul_expression_type<int_constant<2>, sqrt_expression<E> >::type
			>::type,
			typename E::template diff_type<M>::type
		>::type type;
	};

    constexpr sqrt_expression(const expression<E> &e) : e(e()) {}

    constexpr const E& expr() const { return e; }

	template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
		return int_constant<1>() / (int_constant<2>() * sqrt(e)) * e.diff<M>();
	}

	template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
		return std::sqrt(e(vars));
	}

private:
	const E e;
};

template<class E>
constexpr sqrt_expression<E> sqrt(const expression<E>& e) {
	return sqrt_expression<E>(e);
}

_SYMDIFF_END
