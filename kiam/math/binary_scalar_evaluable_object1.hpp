#pragma once

#include "evaluable_object.hpp"
#include "context.hpp"

_KIAM_MATH_BEGIN

template<class EO, class BO>
struct binary_scalar_evaluable_object1 : evaluable_object<typename EO::tag_type, binary_scalar_evaluable_object1<EO, BO> >
{
    typedef BO bin_op_type;
    typedef typename bin_op_type::second_argument_type arg_type;
    typedef typename bin_op_type::result_type value_type;

    binary_scalar_evaluable_object1(const EOBJ(EO) &eobj, const bin_op_type &bin_op, const arg_type &value) :
        eobj_proxy(eobj.get_proxy()), bin_op(bin_op), value(value){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return bin_op(eobj_proxy[i], value);
    }

    __DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
        return bin_op(eobj_proxy(i), value);
    }

    template<typename CONTEXT>
    __DEVICE
    CONSTEXPR value_type operator()(size_t i, const context<typename EO::tag_type, CONTEXT> &context) const {
        return bin_op(eobj_proxy(i, context), value);
    }

private:
    const typename EO::proxy_type eobj_proxy;
    const bin_op_type bin_op;
    const arg_type value;
};

template<class EO>
binary_scalar_evaluable_object1<EO, plus<get_value_type_t<EO> > >
operator+(const EOBJ(EO) &eobj, const get_value_type_t<EO> &value)
{
    typedef plus<get_value_type_t<EO> > bin_op_type;
    return binary_scalar_evaluable_object1<EO, bin_op_type>(eobj, bin_op_type(), value);
}

template<class EO>
binary_scalar_evaluable_object1<EO, minus<get_value_type_t<EO> > >
operator-(const EOBJ(EO) &eobj, const get_value_type_t<EO> &value)
{
    typedef minus<get_value_type_t<EO> > bin_op_type;
    return binary_scalar_evaluable_object1<EO, bin_op_type>(eobj, bin_op_type(), value);
}

template<class EO>
binary_scalar_evaluable_object1<EO,
    generic_multiplies<
        get_value_type_t<EO>,
        get_scalar_type_t<get_value_type_t<EO> >,
        get_value_type_t<EO>
    >
> operator*(const EOBJ(EO) &eobj, const get_scalar_type_t<get_value_type_t<EO> > &value)
{
    typedef generic_multiplies<
        get_value_type_t<EO>,
        get_scalar_type_t<get_value_type_t<EO> >,
        get_value_type_t<EO>
    > bin_op_type;
    return binary_scalar_evaluable_object1<EO, bin_op_type>(eobj, bin_op_type(), value);
}

template<class EO>
binary_scalar_evaluable_object1<EO,
    generic_divides<
        get_value_type_t<EO>,
        get_scalar_type_t<get_value_type_t<EO> >,
        get_value_type_t<EO>
    >
> operator/(const EOBJ(EO) &eobj, const get_scalar_type_t<get_value_type_t<EO> > &value)
{
    typedef generic_divides<
        get_value_type_t<EO>,
        get_scalar_type_t<get_value_type_t<EO> >,
        get_value_type_t<EO>
    > bin_op_type;
    return binary_scalar_evaluable_object1<EO, bin_op_type>(eobj, bin_op_type(), value);
}

_KIAM_MATH_END
