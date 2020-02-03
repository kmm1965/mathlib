#pragma once

#include "div_expression.hpp"

_SYMDIFF_BEGIN

template<class E>
struct log_expression : expression<log_expression<E> >
{
    template<unsigned M>
    struct diff_type
    {
        typedef typename div_expression_type<
            typename E::template diff_type<M>::type,
            E
        >::type type;
    };

    constexpr log_expression(const expression<E> &e) : e(e()){}

    constexpr const E& expr() const { return e; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return e.diff<M>() / e;
    }

    template<typename T, size_t _Size>
    constexpr T operator()(const std::array<T, _Size> &vars) const {
        return std::log(e(vars));
    }

private:
    const E e;
};

template<class E>
constexpr log_expression<E> log(const expression<E>& e){
    return log_expression<E>(e);
}

_SYMDIFF_END
