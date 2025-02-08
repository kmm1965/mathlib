#pragma once

#include "div_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct tan_expression : expression<tan_expression<E> >
{
    typedef div_expression_t<
        typename E::diff_type,
        mul_expression<cos_expression<E>, cos_expression<E> >
    > diff_type;

    constexpr tan_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const
    {
        auto cos_e = cos(e);
        return e.diff() / (cos_e * cos_e);
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return std::tan(e(x));
    }

    constexpr std::string to_string() const {
        return "tan(" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr tan_expression<E> tan(expression<E> const& e){
    return tan_expression<E>(e);
}

_SYMDIFF1_END
