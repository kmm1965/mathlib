#pragma once

#include "mul_expression.hpp"
#include "sign_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct abs_expression : expression<abs_expression<E> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef mul_expression_t<
            sign_expression<E>,
            typename E::template diff_type<M>::type
        > type;
    };

    constexpr abs_expression(expression<E> const& e) : e(e()){}

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return sign(e) * e.diff<M>();
    }

    template<typename T>
    constexpr T operator()(T const& x) const {
        return std::abs(e(x));
    }

private:
    E const e;
};

template<class E>
constexpr abs_expression<E> abs(expression<E> const& e) {
    return abs_expression<E>(e);
}

_SYMDIFF_END
