#pragma once

#include "mul_expression.hpp"
#include "negate_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct sin_expression;

template<class E>
struct cos_expression : expression<cos_expression<E> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef mul_expression_t<
            negate_expression<sin_expression<E> >,
            typename E::template diff_type<M>::type
        > type;
    };

    constexpr cos_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return -sin(e) * e.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(std::array<T, _Size> const& vars) const {
        return std::cos(e(vars));
    }

private:
    E const e;
};

template<class E>
constexpr cos_expression<E> cos(expression<E> const& e){
    return cos_expression<E>(e);
}

_SYMDIFF_END
