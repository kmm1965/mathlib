#pragma once

#include "mul_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct exp_expression : expression<exp_expression<E> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef mul_expression_t<
            exp_expression<E>,
            typename E::template diff_type<M>::type
        > type;
    };

    constexpr exp_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return exp(e) * e.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(std::array<T, _Size> const& vars) const {
        return std::exp(e(vars));
    }

private:
    E const e;
};

template<class E>
constexpr exp_expression<E> exp(expression<E> const& e){
    return exp_expression<E>(e);
}

_SYMDIFF_END
