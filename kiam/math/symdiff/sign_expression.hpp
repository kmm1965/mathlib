#pragma once

#include "int_constant.hpp"

_SYMDIFF_BEGIN

template<class E>
struct sign_expression : expression<sign_expression<E> >
{
	template<unsigned M>
	struct diff_type
	{
		typedef int_constant<0> type;
	};

    constexpr sign_expression(const expression<E> &e) : e(e()) {}

    constexpr const E& expr() const { return e; }

	template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
		return typename diff_type<M>::type();
	}

	template<typename T>
    constexpr int operator()(const T &x) const
	{
		T value = e(x);
		return T < 0 ? -1 : T > 0 ? 1 : 0;
	}

private:
	const E e;
};

template<class E>
constexpr sign_expression<E> sign(const expression<E>& e) {
	return sign_expression<E>(e);
}

_SYMDIFF_END
