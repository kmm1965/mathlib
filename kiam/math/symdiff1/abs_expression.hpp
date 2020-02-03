#pragma once

#include "mul_expression.hpp"
#include "sign_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct abs_expression : expression<abs_expression<E> >
{
    typedef typename mul_expression_type<
        sign_expression<E>,
        typename E::diff_type
    >::type diff_type;

    constexpr abs_expression(const expression<E> &e) : e(e()) {}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
        return sign(e) * e.diff();
    }

    template<typename T>
    constexpr T operator()(const T &x) const {
        return std::abs(e(x));
    }

    constexpr std::string to_string() const {
        return "abs(" + e.to_string() + ')';
    }

private:
    const E e;
};

template<class E>
constexpr abs_expression<E> abs(const expression<E>& e) {
    return abs_expression<E>(e);
}

_SYMDIFF1_END
