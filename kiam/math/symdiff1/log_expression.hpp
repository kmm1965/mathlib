#pragma once

#include "div_expression.hpp"

_SYMDIFF1_BEGIN

template<class E>
struct log_expression : expression<log_expression<E> >
{
    typedef typename div_expression_type<
        typename E::diff_type,
        E
    >::type diff_type;

    constexpr log_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

    constexpr diff_type diff() const {
        return e.diff() / e;
    }

    template<typename T>
    constexpr T operator()(const T &x) const {
        return std::log(e(x));
    }

    constexpr std::string to_string() const {
        return "log(" + e.to_string() + ')';
    }

private:
    const E e;
};

template<class E>
constexpr log_expression<E> log(const expression<E>& e){
    return log_expression<E>(e);
}

_SYMDIFF1_END
