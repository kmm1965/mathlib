#pragma once

#include "div_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct log_expression : expression<log_expression<E> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef div_expression_t<
            typename E::template diff_type<M>::type,
            E
        > type;
    };

    constexpr log_expression(expression<E> const& e) : e(e()){}

    constexpr E const& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return e.diff<M>() / e;
    }

    template<typename T, size_t _Size>
    constexpr T operator()(std::array<T, _Size> const& vars) const {
        return std::log(e(vars));
    }

private:
    E const e;
};

template<class E>
constexpr log_expression<E> log(expression<E> const& e){
    return log_expression<E>(e);
}

_SYMDIFF_END
