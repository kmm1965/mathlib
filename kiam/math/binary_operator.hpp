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

    template<typename T1, typename T2, typename T3>
    struct get_value_type3
    {
        typedef typename OP1::template get_value_type3<T1, T2, T3>::type type;
        static_assert(std::is_same<type, typename OP2::template get_value_type3<T1, T2, T3>::type>::value, "Value types should be the same");
    };

    binary_operator(MATH_OP(OP1) const& op1, MATH_OP(OP2) const& op2) : op1_proxy(op1.get_proxy()), op2_proxy(op2.get_proxy()){}

    template<class GEXP_P>
    __DEVICE
    typename get_value_type<get_value_type_t<GEXP_P>>::type
    operator()(size_t i, GEXP_P const& gexp_proxy) const
    {
        return typename BO::template apply<typename get_value_type<get_value_type_t<GEXP_P>>::type>::type()(
            op1_proxy(i, gexp_proxy), op2_proxy(i, gexp_proxy));
    }

    template<class GEXP1_P, class GEXP2_P>
    __DEVICE
    typename get_value_type<get_value_type_t<GEXP1_P>>::type
    operator()(size_t i, GEXP1_P const& gexp1_proxy, GEXP2_P const& gexp2_proxy) const
    {
        return typename BO::template apply<typename get_value_type2<get_value_type_t<GEXP1_P>, get_value_type_t<GEXP2_P>>::type>::type()(
            op1_proxy(i, gexp1_proxy, gexp2_proxy), op2_proxy(i, gexp1_proxy, gexp2_proxy));
    }

    template<class GEXP1_P, class GEXP2_P, class GEXP3_P>
    __DEVICE
    typename get_value_type<get_value_type_t<GEXP1_P>>::type
    operator()(size_t i, GEXP1_P const& gexp1_proxy, GEXP2_P const& gexp2_proxy, GEXP3_P const& gexp3_proxy) const
    {
        return typename BO::template apply<typename get_value_type3<get_value_type_t<GEXP1_P>, get_value_type_t<GEXP2_P>, get_value_type_t<GEXP3_P>>::type>::type()(
            op1_proxy(i, gexp1_proxy, gexp2_proxy, gexp3_proxy), op2_proxy(i, gexp1_proxy, gexp2_proxy, gexp3_proxy));
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(binary_operator)
    IMPLEMENT_MATH_EVAL_OPERATOR_N(2, binary_operator)
    IMPLEMENT_MATH_EVAL_OPERATOR_N(3, binary_operator)

private:
    typename OP1::proxy_type const op1_proxy;
    typename OP2::proxy_type const op2_proxy;
};

#define BINARY_GRID_OP(oper, bin_op) \
    template<class OP1, class OP2> \
    binary_operator<OP1, bin_op, OP2> \
    operator oper(MATH_OP(OP1) const& op1, MATH_OP(OP2) const& op2){ \
        return binary_operator<OP1, bin_op, OP2>(op1, op2); \
    }

BINARY_GRID_OP(+, math_plus_value)
BINARY_GRID_OP(-, math_minus_value)
BINARY_GRID_OP(*, math_multiplies_value)
BINARY_GRID_OP(/, math_divides_value)

_KIAM_MATH_END
