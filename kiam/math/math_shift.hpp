#pragma once

#include "math_operator.hpp"

_KIAM_MATH_BEGIN

template<typename TAG>
struct math_shift : math_operator<TAG, math_shift<TAG> >
{
    math_shift(int s) : s(s){}

    template<class GEXP_P, typename CONTEXT>
    __DEVICE
    get_value_type_t<GEXP_P>
    operator()(size_t i, const GEXP_P &gexp_proxy, context<TAG, CONTEXT> const& context) const {
        static_assert(std::is_same<TAG, typename GEXP_P::tag_type>::value, "Tag types should be the same");
        return gexp_proxy(i + s, context);
    }

    template<class GEXP_P>
    __DEVICE
    get_value_type_t<GEXP_P>
    operator()(size_t i, const GEXP_P &gexp_proxy) const {
        static_assert(std::is_same<TAG, typename GEXP_P::tag_type>::value, "Tag types should be the same");
        return gexp_proxy[i + s];
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(math_shift)

private:
    int s;
};

template<class GEXP>
operator_evaluator<math_shift<typename GEXP::tag_type>, GEXP> operator>>(GRID_EXPR(GEXP) const& gexp, int s){
    return math_shift<typename GEXP::tag_type>(s)(gexp);
}

template<class GEXP>
operator_evaluator<math_shift<typename GEXP::tag_type>, GEXP> operator<<(GRID_EXPR(GEXP) const& gexp, int s){
    return math_shift<typename GEXP::tag_type>(-s)(gexp);
}

_KIAM_MATH_END
