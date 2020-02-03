#pragma once

#include "mul_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct cos_expression;

template<class E>
struct sin_expression : expression<sin_expression<E> >
{
    typedef typename mul_expression_type<
        cos_expression<E>,
        typename E::diff_type
    >::type diff_type;

    constexpr sin_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
        return cos(e) * e.diff();
    }

    template<typename T>
    constexpr T operator()(const T &x) const {
        return std::sin(e(x));
    }

    constexpr std::string to_string() const {
        return "sin(" + e.to_string() + ')';
    }

private:
    const E e;
};

template<class E>
constexpr sin_expression<E> sin(const expression<E>& e){
    return sin_expression<E>(e);
}

_SYMDIFF1_END
