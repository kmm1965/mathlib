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
        typedef typename mul_expression_type<
            cos_expression<E>,
            typename E::template diff_type<M>::type
        >::type type;
    };

    constexpr sin_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return cos(e) * e.diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
        return std::sin(e(vars));
    }

private:
    const E e;
};

template<class E>
constexpr sin_expression<E> sin(const expression<E>& e){
    return sin_expression<E>(e);
}

_SYMDIFF_END
