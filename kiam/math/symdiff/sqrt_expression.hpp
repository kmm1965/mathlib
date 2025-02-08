#pragma once

#include "div_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct sqrt_expression : expression<sqrt_expression<E> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef mul_expression_t<
            typename div_expression_t<
                int_constant<1>,
                mul_expression_t<int_constant<2>, sqrt_expression<E> >
            >,
            typename E::template diff_type<M>::type
        > type;
    };

    constexpr sqrt_expression(expression<E> const& e) : e(e()) {}

    constexpr E const& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return int_constant<1>() / (int_constant<2>() * sqrt(e)) * e.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
        return std::sqrt(e(vars));
    }

private:
    E const e;
};

template<class E>
constexpr sqrt_expression<E> sqrt(expression<E> const& e){
    return sqrt_expression<E>(e);
}

_SYMDIFF_END
