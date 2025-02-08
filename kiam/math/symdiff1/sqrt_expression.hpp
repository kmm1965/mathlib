#pragma once

#include "div_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct sqrt_expression : expression<sqrt_expression<E> >
{
    typedef mul_expression_t<
        div_expression_t<
            int_constant<1>,
            mul_expression_t<int_constant<2>, sqrt_expression<E> >
        >,
        typename E::diff_type
    > diff_type;

    constexpr sqrt_expression(expression<E> const& e) : e(e()) {}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const {
        return int_constant<1>() / (int_constant<2>() * sqrt(e)) * e.diff();
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return std::sqrt(e(x));
    }

    constexpr std::string to_string() const {
        return "sqrt(" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr sqrt_expression<E> sqrt(expression<E> const& e) {
    return sqrt_expression<E>(e);
}

_SYMDIFF1_END
