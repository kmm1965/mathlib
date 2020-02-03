#pragma once

#include "int_constant.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct sign_expression : expression<sign_expression<E> >
{
    typedef int_constant<0> diff_type;

    constexpr sign_expression(const expression<E> &e) : e(e()) {}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
        return diff_type();
    }

    template<typename T>
    constexpr int operator()(const T &x) const
    {
        T value = e(x);
        return T < 0 ? -1 : T > 0 ? 1 : 0;
    }

    constexpr std::string to_string() const {
        return "sign(" + e.to_string() + ')';
    }

private:
    const E e;
};

template<class E>
constexpr sign_expression<E> sign(const expression<E>& e) {
    return sign_expression<E>(e);
}

_SYMDIFF1_END
