#pragma once

#include "int_constant.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct sign_expression : expression<sign_expression<E> >
{
    typedef int_constant<0> diff_type;

    constexpr sign_expression(expression<E> const& e) : e(e()) {}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const {
        return diff_type();
    }

    template<typename T>
    constexpr int operator()(T const& x) const
    {
        T value = e(x);
        return T < 0 ? -1 : T > 0 ? 1 : 0;
    }

    constexpr std::string to_string() const {
        return "sign(" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr sign_expression<E> sign(expression<E> const& e) {
    return sign_expression<E>(e);
}

_SYMDIFF1_END
