#pragma once

#include "mul_expression.hpp"
#include "sign_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct abs_expression : expression<abs_expression<E> >
{
    typedef mul_expression_t<
        sign_expression<E>,
        typename E::diff_type
    > diff_type;

    constexpr abs_expression(expression<E> const& e) : e(e()) {}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const {
        return sign(e) * e.diff();
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return std::abs(e(x));
    }

    constexpr std::string to_string() const {
        return "abs(" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr abs_expression<E> abs(expression<E> const& e) {
    return abs_expression<E>(e);
}

_SYMDIFF1_END
