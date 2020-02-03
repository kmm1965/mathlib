#pragma once

#include "math_operator.hpp"

_KIAM_MATH_BEGIN

template<typename TAG>
struct math_shift : math_operator<TAG, math_shift<TAG> >
{
    math_shift(int s) : s(s){}

    template<class EOP, class CONTEXT>
    __DEVICE
    get_value_type_t<EOP>
    operator()(size_t i, const EOP &eobj_proxy, const context<TAG, CONTEXT> &context) const {
        static_assert(std::is_same<TAG, typename EOP::tag_type>::value, "Tag types should be the same");
        return eobj_proxy(i + s, context);
    }

    template<class EOP>
    __DEVICE
    get_value_type_t<EOP>
    operator()(size_t i, const EOP &eobj_proxy) const {
        static_assert(std::is_same<TAG, typename EOP::tag_type>::value, "Tag types should be the same");
        return eobj_proxy[i + s];
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(math_shift)

private:
    int s;
};

template<class EO>
operator_evaluator<math_shift<typename EO::tag_type>, EO> operator>>(
    const EOBJ(EO) &eobj,
    int s
){
    return math_shift<typename EO::tag_type>(s)(eobj);
}

template<class EO>
operator_evaluator<math_shift<typename EO::tag_type>, EO> operator<<(
    const EOBJ(EO) &eobj,
    int s
){
    return math_shift<typename EO::tag_type>(-s)(eobj);
}

_KIAM_MATH_END
