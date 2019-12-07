#pragma once

#include "mul_expression.hpp"
#include "sign_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct abs_expression : expression<abs_expression<E> >
{
	template<unsigned M>
	struct diff_type
	{
		typedef typename mul_expression_type<
			sign_expression<E>,
			typename E::template diff_type<M>::type
		>::type type;
	};

	constexpr abs_expression(const expression<E> &e) : e(e()){}

	template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
		return sign(e) * e.diff<M>();
	}

	template<typename T>
    constexpr T operator()(const T &x) const {
		return std::abs(e(x));
	}

private:
	const E e;
};

template<class E>
constexpr abs_expression<E> abs(const expression<E>& e) {
	return abs_expression<E>(e);
}

_SYMDIFF_END
