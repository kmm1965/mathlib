#pragma once

#include "div_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct tan_expression : expression<tan_expression<E> >
{
	template<unsigned M>
	struct diff_type
	{
		typedef typename div_expression_type<
			typename E::template diff_type<M>::type,
			mul_expression<cos_expression<E>, cos_expression<E> >
		>::type type;
	};

    constexpr tan_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

	template<unsigned M>
    constexpr typename diff_type<M>::type diff() const
	{
		auto cos_e = cos(e);
		return e.diff<M>() / (cos_e * cos_e);
	}

	template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
		return std::tan(e(vars));
	}

private:
	const E e;
};

template<class E>
constexpr tan_expression<E> tan(const expression<E>& e){
	return tan_expression<E>(e);
}

_SYMDIFF_END
