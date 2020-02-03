#pragma once

#include "math_operator.hpp"
#include "generic_operators.hpp"

_KIAM_MATH_BEGIN

template<class OP1, class BO, class OP2>
struct binary_operator : math_operator<binary_operator<OP1, BO, OP2> >
{
    typedef math_operator<binary_operator> super;

    template<typename EO_TAG>
    struct get_tag_type
    {
        typedef typename OP1::template get_tag_type<EO_TAG>::type type;
        static_assert(std::is_same<type, typename OP2::template get_tag_type<EO_TAG>::type>::value, "Tag types should be the same");
    };

#define MATH_OPERATOR_GET_TAG(z, n, unused) \
    template<BOOST_PP_ENUM_PARAMS(n, class EO_TAG)> \
    struct get_tag_type##n \
    { \
        typedef typename OP1::template get_tag_type##n<BOOST_PP_ENUM_PARAMS(n, EO_TAG)>::type type; \
        static_assert(std::is_same<type, typename OP2::template get_tag_type##n<BOOST_PP_ENUM_PARAMS(n, EO_TAG)>::type>::value, "Tag types should be the same"); \
    };
    BOOST_PP_REPEAT_FROM_TO(2, BOOST_PP_INC(MAX_MATH_OPERATOR_PARAMS), MATH_OPERATOR_GET_TAG, ~)
#undef MATH_OPERATOR_GET_TAG

    template<typename T>
    struct get_value_type
    {
        typedef typename OP1::template get_value_type<T>::type type;
        static_assert(std::is_same<type, typename OP2::template get_value_type<T>::type>::value, "Value types should be the same");
    };

    template<typename T1, typename T2>
    struct get_value_type2
    {
        typedef typename OP1::template get_value_type2<T1, T2>::type type;
        static_assert(std::is_same<type, typename OP2::template get_value_type2<T1, T2>::type>::value, "Value types should be the same");
    };

    binary_operator(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2) : op1_proxy(op1.get_proxy()), op2_proxy(op2.get_proxy()){}

    template<class EOP>
    __DEVICE
    typename get_value_type<get_value_type_t<EOP>>::type
    CONSTEXPR operator()(size_t i, const EOP &eobj_proxy) const
    {
        return typename BO::template apply<typename get_value_type<get_value_type_t<EOP>>::type>::type()(
            op1_proxy(i, eobj_proxy), op2_proxy(i, eobj_proxy));
    }

    template<class EOP1, class EOP2>
    __DEVICE
    typename get_value_type<get_value_type_t<EOP1>>::type
    CONSTEXPR operator()(size_t i, const EOP1 &eobj1_proxy, const EOP2 &eobj2_proxy) const
    {
        return typename BO::template apply<typename get_value_type2<get_value_type_t<EOP1>, get_value_type_t<EOP2>>::type>::type()(
            op1_proxy(i, eobj1_proxy, eobj2_proxy), op2_proxy(i, eobj1_proxy, eobj2_proxy));
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(binary_operator)
    IMPLEMENT_MATH_EVAL_OPERATOR2(binary_operator)

private:
    const typename OP1::proxy_type op1_proxy;
    const typename OP2::proxy_type op2_proxy;
};

template<class OP1, class OP2>
binary_operator<OP1, math_plus_value, OP2>
operator+(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2){
    return binary_operator<OP1, math_plus_value, OP2>(op1, op2);
}

template<class OP1, class OP2>
binary_operator<OP1, math_minus_value, OP2>
operator-(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2){
    return binary_operator<OP1, math_minus_value, OP2>(op1, op2);
}

template<class OP1, class OP2>
binary_operator<OP1, math_multiplies_value, OP2>
operator*(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2){
    return binary_operator<OP1, math_multiplies_value, OP2>(op1, op2);
}

template<class OP1, class OP2>
binary_operator<OP1, math_divides_value, OP2>
operator/(const MATH_OP(OP1) &op1, const MATH_OP(OP2) &op2){
    return binary_operator<OP1, math_divides_value, OP2>(op1, op2);
}

_KIAM_MATH_END
