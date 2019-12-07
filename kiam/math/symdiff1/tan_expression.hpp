#pragma once

#include "div_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct tan_expression : expression<tan_expression<E> >
{
	typedef typename div_expression_type<
		typename E::diff_type,
		mul_expression<cos_expression<E>, cos_expression<E> >
	>::type diff_type;

    constexpr tan_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const
	{
		auto cos_e = cos(e);
		return e.diff() / (cos_e * cos_e);
	}

	template<typename T>
    constexpr T operator()(const T &x) const {
		return std::tan(e(x));
	}

    constexpr std::string to_string() const {
        return "tan(" + e.to_string() + ')';
    }

private:
	const E e;
};

template<class E>
constexpr tan_expression<E> tan(const expression<E>& e){
	return tan_expression<E>(e);
}

_SYMDIFF1_END
