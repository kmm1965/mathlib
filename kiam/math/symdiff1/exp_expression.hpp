#pragma once

#include "mul_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct exp_expression : expression<exp_expression<E> >
{
    typedef mul_expression_t<
        exp_expression<E>,
        typename E::diff_type
    > diff_type;

    constexpr exp_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const {
        return exp(e) * e.diff();
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return std::exp(e(x));
    }

    constexpr std::string to_string() const {
        return "sin(" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr exp_expression<E> exp(expression<E> const& e){
    return exp_expression<E>(e);
}

_SYMDIFF1_END
