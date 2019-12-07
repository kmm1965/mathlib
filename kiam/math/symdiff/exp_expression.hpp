#pragma once

#include "mul_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct exp_expression : expression<exp_expression<E> >
{
	template<unsigned M>
	struct diff_type
	{
		typedef typename mul_expression_type<
			exp_expression<E>,
			typename E::template diff_type<M>::type
		>::type type;
	};

    constexpr exp_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

	template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
		return exp(e) * e.diff<M>();
	}

	template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
		return std::exp(e(vars));
	}

private:
	const E e;
};

template<class E>
constexpr exp_expression<E> exp(const expression<E>& e){
	return exp_expression<E>(e);
}

_SYMDIFF_END
