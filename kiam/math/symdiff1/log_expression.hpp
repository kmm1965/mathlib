#pragma once

#include "div_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct log_expression : expression<log_expression<E> >
{
    typedef div_expression_t<
        typename E::diff_type,
        E
    > diff_type;

    constexpr log_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    constexpr diff_type diff() const {
        return e.diff() / e;
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return std::log(e(x));
    }

    constexpr std::string to_string() const {
        return "log(" + e.to_string() + ')';
    }

private:
    E const e;
};

template<class E>
constexpr log_expression<E> log(expression<E> const& e){
    return log_expression<E>(e);
}

_SYMDIFF1_END
