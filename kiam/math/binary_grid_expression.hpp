#pragma once

#include "grid_expression.hpp"
#include "context.hpp"

_KIAM_MATH_BEGIN

template<class EO1, class BO, class EO2>
struct binary_grid_expression : grid_expression<typename EO1::tag_type, binary_grid_expression<EO1, BO, EO2> >
{
    static_assert(std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value, "Tag types should be the same");

    typedef BO bin_op_type;
    typedef typename bin_op_type::result_type value_type;

    binary_grid_expression(GRID_EXPR(EO1) const& gexp1, bin_op_type const& bin_op, GRID_EXPR(EO2) const& gexp2) :
        gexp1_proxy(gexp1.get_proxy()), bin_op(bin_op), gexp2_proxy(gexp2.get_proxy()){}

    __DEVICE
    constexpr value_type operator[](size_t i) const {
        return bin_op(gexp1_proxy[i], gexp2_proxy[i]);
    }

    __DEVICE
    constexpr value_type operator()(size_t i) const {
        return bin_op(gexp1_proxy(i), gexp2_proxy(i));
    }

    template<typename CONTEXT>
    __DEVICE
    constexpr value_type operator()(size_t i, context<CONTEXT> const& context) const {
        return bin_op(gexp1_proxy(i, context), gexp2_proxy(i, context));
    }

private:
    typename EO1::proxy_type const gexp1_proxy;
    bin_op_type const bin_op;
    typename EO2::proxy_type const gexp2_proxy;
};

#define DECLARE_EVAL_OBJ_MATH_OPERATION(oper, bin_op) \
    template<class EO1, class EO2> \
    typename std::enable_if< \
        std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value && \
        std::is_same<get_value_type_t<EO1>, get_value_type_t<EO2> >::value, \
        binary_grid_expression<EO1, bin_op<get_value_type_t<EO1> >, EO2> \
    >::type operator oper(GRID_EXPR(EO1) const& gexp1, GRID_EXPR(EO2) const& gexp2) { \
        typedef bin_op<get_value_type_t<EO1> > bin_op_type; \
        return binary_grid_expression<EO1, bin_op_type, EO2>(gexp1, bin_op_type(), gexp2); \
    }

DECLARE_EVAL_OBJ_MATH_OPERATION(+, math_plus)
DECLARE_EVAL_OBJ_MATH_OPERATION(-, math_minus)

template<class EO1, class EO2>
typename std::enable_if<
    std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value &&
        supports_multiplies<get_value_type_t<EO1>, get_value_type_t<EO2> >::value,
    binary_grid_expression<EO1, get_generic_multiplies<get_value_type_t<EO1>, get_value_type_t<EO2> >, EO2>
>::type
operator*(GRID_EXPR(EO1) const& gexp1, GRID_EXPR(EO2) const& gexp2){
    typedef get_generic_multiplies<get_value_type_t<EO1>, get_value_type_t<EO2> > bin_op_type;
    return binary_grid_expression<EO1, bin_op_type, EO2>(gexp1, bin_op_type(), gexp2);
}

template<class EO1, class EO2>
typename std::enable_if<
    std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value &&
        supports_divides<get_value_type_t<EO1>, get_value_type_t<EO2> >::value,
    binary_grid_expression<EO1, get_generic_divides<get_value_type_t<EO1>, get_value_type_t<EO2> >, EO2>
>::type operator/(GRID_EXPR(EO1) const& gexp1, GRID_EXPR(EO2) const& gexp2){
    typedef get_generic_divides<get_value_type_t<EO1>, get_value_type_t<EO2> > bin_op_type;
    return binary_grid_expression<EO1, bin_op_type, EO2>(gexp1, bin_op_type(), gexp2);
}

template<class EO1, class EO2>
typename std::enable_if<
    std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value &&
    std::is_same<get_value_type_t<EO1>, get_value_type_t<EO2> >::value &&
    supports_scalar_product<get_value_type_t<EO1> >::value,
    binary_grid_expression<EO1, generic_scalar_product<get_value_type_t<EO1> >, EO2>
>::type operator&(GRID_EXPR(EO1) const& gexp1, GRID_EXPR(EO2) const& gexp2){
    typedef generic_scalar_product<get_value_type_t<EO1> > bin_op_type;
    return binary_grid_expression<EO1, bin_op_type, EO2>(gexp1, bin_op_type(), gexp2);
}

template<class EO1, class EO2>
typename std::enable_if<
    std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value &&
    std::is_same<get_value_type_t<EO1>, get_value_type_t<EO2> >::value &&
    supports_component_product<get_value_type_t<EO1> >::value,
    binary_grid_expression<EO1, generic_component_product<get_value_type_t<EO1> >, EO2>
>::type operator^(GRID_EXPR(EO1) const& gexp1, GRID_EXPR(EO2) const& gexp2){
    typedef generic_component_product<get_value_type_t<EO1> > bin_op_type;
    return binary_grid_expression<EO1, bin_op_type, EO2>(gexp1, bin_op_type(), gexp2);
}

_KIAM_MATH_END
