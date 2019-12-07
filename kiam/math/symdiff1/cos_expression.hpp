#pragma once

#include "mul_expression.hpp"
#include "negate_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct sin_expression;

template<class E>
struct cos_expression : expression<cos_expression<E> >
{
	typedef typename mul_expression_type<
		negate_expression<sin_expression<E> >,
		typename E::diff_type
	>::type diff_type;

    constexpr cos_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
		return -sin(e) * e.diff();
	}

    template<typename T>
    constexpr T operator()(const T &x) const {
		return std::cos(e(x));
	}

    constexpr std::string to_string() const {
        return "cos(" + e.to_string() + ')';
    }

private:
	const E e;
};

template<class E>
constexpr cos_expression<E> cos(const expression<E>& e){
	return cos_expression<E>(e);
}

_SYMDIFF1_END
