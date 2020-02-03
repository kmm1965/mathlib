#pragma once

#include "mul_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct exp_expression : expression<exp_expression<E> >
{
    typedef typename mul_expression_type<
        exp_expression<E>,
        typename E::diff_type
    >::type diff_type;

    constexpr exp_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
        return exp(e) * e.diff();
    }

    template<typename T>
    constexpr T operator()(const T &x) const {
        return std::exp(e(x));
    }

    constexpr std::string to_string() const {
        return "sin(" + e.to_string() + ')';
    }

private:
    const E e;
};

template<class E>
constexpr exp_expression<E> exp(const expression<E>& e){
    return exp_expression<E>(e);
}

_SYMDIFF1_END
