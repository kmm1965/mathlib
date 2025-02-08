#pragma once

#include "math_operator.hpp"

_KIAM_MATH_BEGIN

template<class MOP, class GEXP_P>
struct composition_operator_gexp_proxy
{
    typedef typename MOP::template get_value_type<get_value_type_t<GEXP_P>>::type value_type;
    typedef typename MOP::template get_tag_type<typename GEXP_P::tag_type>::type tag_type;

    __DEVICE
    CONSTEXPR composition_operator_gexp_proxy(const MOP &op_proxy, const GEXP_P &gexp_proxy) : op_proxy(op_proxy), gexp_proxy(gexp_proxy){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return op_proxy(i, gexp_proxy);
    }

    __DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
        return op_proxy(i, gexp_proxy);
    }

private:
    const MOP &op_proxy;
    const GEXP_P &gexp_proxy;
};

template<class OP1, class OP2>
struct composition_operator : math_operator<composition_operator<OP1, OP2> >
{
    template<typename EO_TAG>
    struct get_tag_type
    {
        typedef typename OP1::template get_tag_type<typename OP2::template get_tag_type<EO_TAG>::type>::type type;
    };

    template<class T>
    struct get_value_type
    {
        typedef typename OP1::template get_value_type<
            typename OP2::template get_value_type<T>::type
        >::type type;
    };

    composition_operator(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2) :
        op1_proxy(op1.get_proxy()), op2_proxy(op2.get_proxy()){}

    template<class GEXP_P>
    __DEVICE
    typename get_value_type<get_value_type_t<GEXP_P>>::type
    CONSTEXPR operator()(size_t i, const GEXP_P &gexp_proxy) const {
        return op1_proxy(i, composition_operator_gexp_proxy<typename OP2::proxy_type, GEXP_P>(op2_proxy, gexp_proxy));
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(composition_operator)

private:
    typename OP1::proxy_type const op1_proxy;
    typename OP2::proxy_type const op2_proxy;
};

template<class OP1, class OP2>
composition_operator<OP1, OP2> operator&(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2){
    return composition_operator<OP1, OP2>(op1, op2);
}

_KIAM_MATH_END
