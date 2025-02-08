#pragma once

#include "mul_expression.hpp"
#include "negate_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct sin_expression;

template<class E>
struct cos_expression : expression<cos_expression<E> >
{
    typedef mul_expression_t<
        negate_expression<sin_expression<E> >,
        typename E::diff_type
    > diff_type;

    constexpr cos_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const {
        return -sin(e) * e.diff();
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return std::cos(e(x));
    }

    constexpr std::string to_string() const {
        return "cos(" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr cos_expression<E> cos(expression<E> const& e){
    return cos_expression<E>(e);
}

_SYMDIFF1_END
