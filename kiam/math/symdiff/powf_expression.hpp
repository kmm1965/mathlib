#pragma once

#include "mul_expression.hpp"
#include "scalar.hpp"

_SYMDIFF_BEGIN

template<class E>
struct powf_expression : expression<powf_expression<E> >
{
    typedef powf_expression type;

    template<unsigned M>
    struct diff_type {
        typedef mul_expression_t<
            mul_expression<
                scalar<double>,
                powf_expression<E>
            >,
            typename E::template diff_type<M>::type
        > type;
    };

    constexpr powf_expression(expression<E> const& e, double pow) : e(e()), pow_(pow) {}

    constexpr E const& expr() const { return e; }
    constexpr double pow() const { return pow_; }

    template<unsigned M>
    constexpr typename diff_type<M>::type diff() const {
        return _(pow_) * kiam::math::symdiff::powf(e, pow_ - 1) * e.template diff<M>();
    }

    template<typename T, size_t _Size>
    constexpr T operator()(std::array<T, _Size> const& vars) const {
        return _KIAM_MATH::pow_(e(vars), pow_);
    }

private:
    E const e;
    double const pow_;
};

template<class E>
constexpr powf_expression<E> powf(expression<E> const& e, double pow) {
    return powf_expression<E>(e, pow);
}

_SYMDIFF_END
