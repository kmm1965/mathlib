#pragma once

#include "evaluable_object.hpp"
#include "context.hpp"

_KIAM_MATH_BEGIN

template<class EO1, class BO, class EO2>
struct binary_evaluable_object : evaluable_object<typename EO1::tag_type, binary_evaluable_object<EO1, BO, EO2> >
{
    static_assert(std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value, "Tag types should be the same");

    typedef BO bin_op_type;
    typedef typename bin_op_type::result_type value_type;

    binary_evaluable_object(const EOBJ(EO1) &eobj1, const bin_op_type &bin_op, const EOBJ(EO2) &eobj2) :
        eobj1_proxy(eobj1.get_proxy()), bin_op(bin_op), eobj2_proxy(eobj2.get_proxy()){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return bin_op(eobj1_proxy[i], eobj2_proxy[i]);
    }

    __DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
        return bin_op(eobj1_proxy(i), eobj2_proxy(i));
    }

    template<typename CONTEXT>
    __DEVICE
    CONSTEXPR value_type operator()(size_t i, const context<typename EO1::tag_type, CONTEXT> &context) const {
        return bin_op(eobj1_proxy(i, context), eobj2_proxy(i, context));
    }

private:
    const typename EO1::proxy_type eobj1_proxy;
    const bin_op_type bin_op;
    const typename EO2::proxy_type eobj2_proxy;
};

#define DECLARE_EVAL_OBJ_MATH_OPERATION(oper, bin_op) \
    template<class EO1, class EO2> \
    typename std::enable_if< \
        std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value && \
        std::is_same<get_value_type_t<EO1>, get_value_type_t<EO2> >::value, \
        binary_evaluable_object<EO1, bin_op<get_value_type_t<EO1> >, EO2> \
    >::type operator oper(const EOBJ(EO1)& eobj1, const EOBJ(EO2)& eobj2) { \
        typedef bin_op<get_value_type_t<EO1> > bin_op_type; \
        return binary_evaluable_object<EO1, bin_op_type, EO2>(eobj1, bin_op_type(), eobj2); \
    }

DECLARE_EVAL_OBJ_MATH_OPERATION(+, plus)
DECLARE_EVAL_OBJ_MATH_OPERATION(-, minus)

template<class EO1, class EO2>
typename std::enable_if<
    std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value &&
        supports_multiplies<get_value_type_t<EO1>, get_value_type_t<EO2> >::value,
    binary_evaluable_object<EO1, get_generic_multiplies<get_value_type_t<EO1>, get_value_type_t<EO2> >, EO2>
>::type
operator*(const EOBJ(EO1) &eobj1, const EOBJ(EO2) &eobj2){
    typedef get_generic_multiplies<get_value_type_t<EO1>, get_value_type_t<EO2> > bin_op_type;
    return binary_evaluable_object<EO1, bin_op_type, EO2>(eobj1, bin_op_type(), eobj2);
}

template<class EO1, class EO2>
typename std::enable_if<
    std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value &&
        supports_divides<get_value_type_t<EO1>, get_value_type_t<EO2> >::value,
    binary_evaluable_object<EO1, get_generic_divides<get_value_type_t<EO1>, get_value_type_t<EO2> >, EO2>
>::type operator/(const EOBJ(EO1) &eobj1, const EOBJ(EO2) &eobj2){
    typedef get_generic_divides<get_value_type_t<EO1>, get_value_type_t<EO2> > bin_op_type;
    return binary_evaluable_object<EO1, bin_op_type, EO2>(eobj1, bin_op_type(), eobj2);
}

template<class EO1, class EO2>
typename std::enable_if<
    std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value &&
    std::is_same<get_value_type_t<EO1>, get_value_type_t<EO2> >::value &&
    supports_scalar_product<get_value_type_t<EO1> >::value,
    binary_evaluable_object<EO1, generic_scalar_product<get_value_type_t<EO1> >, EO2>
>::type operator&(const EOBJ(EO1) &eobj1, const EOBJ(EO2) &eobj2){
    typedef generic_scalar_product<get_value_type_t<EO1> > bin_op_type;
    return binary_evaluable_object<EO1, bin_op_type, EO2>(eobj1, bin_op_type(), eobj2);
}

template<class EO1, class EO2>
typename std::enable_if<
    std::is_same<typename EO1::tag_type, typename EO2::tag_type>::value &&
    std::is_same<get_value_type_t<EO1>, get_value_type_t<EO2> >::value &&
    supports_component_product<get_value_type_t<EO1> >::value,
    binary_evaluable_object<EO1, generic_component_product<get_value_type_t<EO1> >, EO2>
>::type operator^(const EOBJ(EO1) &eobj1, const EOBJ(EO2) &eobj2){
    typedef generic_component_product<get_value_type_t<EO1> > bin_op_type;
    return binary_evaluable_object<EO1, bin_op_type, EO2>(eobj1, bin_op_type(), eobj2);
}

_KIAM_MATH_END
