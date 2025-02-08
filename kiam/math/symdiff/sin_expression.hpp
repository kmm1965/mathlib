#pragma once

#include "mul_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct cos_expression;

template<class E>
struct sin_expression : expression<sin_expression<E> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef mul_expression_t<
            cos_expression<E>,
            typename E::template diff_type<M>::type
        > type;
    };

    constexpr sin_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return cos(e) * e.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
        return std::sin(e(vars));
    }

private:
    E const e;
};

template<class E>
constexpr sin_expression<E> sin(expression<E> const& e){
    return sin_expression<E>(e);
}

_SYMDIFF_END
