#pragma once

#include "mul_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct cos_expression;

template<class E>
struct sin_expression : expression<sin_expression<E> >
{
    typedef mul_expression_t<
        cos_expression<E>,
        typename E::diff_type
    > diff_type;

    constexpr sin_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const {
        return cos(e) * e.diff();
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return std::sin(e(x));
    }

    constexpr std::string to_string() const {
        return "sin(" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr sin_expression<E> sin(expression<E> const& e){
    return sin_expression<E>(e);
}

_SYMDIFF1_END
