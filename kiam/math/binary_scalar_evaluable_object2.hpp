#pragma once

#include "evaluable_object.hpp"
#include "context.hpp"

_KIAM_MATH_BEGIN

template<class EO, class BO>
struct binary_scalar_evaluable_object2 : evaluable_object<typename EO::tag_type, binary_scalar_evaluable_object2<EO, BO> >
{
    typedef BO bin_op_type;
    typedef typename bin_op_type::first_argument_type arg_type;
    typedef typename bin_op_type::result_type value_type;

    binary_scalar_evaluable_object2(const arg_type &value, const bin_op_type &bin_op, const EOBJ(EO) &eobj) :
        value(value), bin_op(bin_op), eobj_proxy(eobj.get_proxy()){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return bin_op(value, eobj_proxy[i]);
    }

    __DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
        return bin_op(value, eobj_proxy(i));
    }

    template<class CONTEXT>
    __DEVICE
    CONSTEXPR value_type operator()(size_t i, const context<typename EO::tag_type, CONTEXT> &context) const {
        return bin_op(value, eobj_proxy(i, context));
    }

private:
    const arg_type value;
    const bin_op_type bin_op;
    const typename EO::proxy_type eobj_proxy;
};

template<class EO>
binary_scalar_evaluable_object2<EO, plus<get_value_type_t<EO> > >
operator+(const get_value_type_t<EO> &value, const EOBJ(EO) &eobj)
{
    typedef plus<get_value_type_t<EO> > bin_op_type;
    return binary_scalar_evaluable_object2<EO, bin_op_type>(value, bin_op_type(), eobj);
}

template<class EO>
binary_scalar_evaluable_object2<EO, minus<get_value_type_t<EO> > >
operator-(const get_value_type_t<EO> &value, const EOBJ(EO) &eobj)
{
    typedef minus<get_value_type_t<EO> > bin_op_type;
    return binary_scalar_evaluable_object2<EO, bin_op_type>(value, bin_op_type(), eobj);
}

template<class EO>
binary_scalar_evaluable_object2<EO,
    generic_multiplies<
        get_scalar_type_t<get_value_type_t<EO> >,
        get_value_type_t<EO>,
        get_value_type_t<EO>
    >
> operator*(const get_scalar_type_t<get_value_type_t<EO> > &value, const EOBJ(EO) &eobj)
{
    typedef generic_multiplies<
        get_scalar_type_t<get_value_type_t<EO> >,
        get_value_type_t<EO>,
        get_value_type_t<EO>
    > bin_op_type;
    return binary_scalar_evaluable_object2<EO, bin_op_type>(value, bin_op_type(), eobj);
}

template<class EO>
typename std::enable_if<
    supports_divides<get_scalar_type_t<get_value_type_t<EO> >, get_value_type_t<EO> >::value,
    binary_scalar_evaluable_object2<EO,
        generic_divides<
            get_scalar_type_t<get_value_type_t<EO> >,
            get_value_type_t<EO>,
            get_value_type_t<EO>
        >
    >
>::type operator/(const get_scalar_type_t<get_value_type_t<EO> >&value, const EOBJ(EO) &eobj)
{
    typedef generic_divides<
        get_scalar_type_t<get_value_type_t<EO> >,
        get_value_type_t<EO>,
        get_value_type_t<EO>
    > bin_op_type;
    return binary_scalar_evaluable_object2<EO, bin_op_type>(value, bin_op_type(), eobj);
}

_KIAM_MATH_END
