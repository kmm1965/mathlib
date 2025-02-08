#pragma once

#include "grid_expression.hpp"
#include "context.hpp"

_KIAM_MATH_BEGIN

template<class GEXP, class BO>
struct binary_scalar_grid_expression1 : grid_expression<typename GEXP::tag_type, binary_scalar_grid_expression1<GEXP, BO> >
{
    typedef BO bin_op_type;
    typedef typename bin_op_type::second_argument_type arg_type;
    typedef typename bin_op_type::result_type value_type;

    binary_scalar_grid_expression1(GRID_EXPR(GEXP) const& gexp, bin_op_type const& bin_op, arg_type const& value) :
        gexp_proxy(gexp.get_proxy()), bin_op(bin_op), value(value){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return bin_op(gexp_proxy[i], value);
    }

    __DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
        return bin_op(gexp_proxy(i), value);
    }

    template<typename CONTEXT>
    __DEVICE
    CONSTEXPR value_type operator()(size_t i, context<CONTEXT> const& context) const {
        return bin_op(gexp_proxy(i, context), value);
    }

private:
    typename GEXP::proxy_type const gexp_proxy;
    bin_op_type const bin_op;
    arg_type const value;
};

template<class GEXP>
binary_scalar_grid_expression1<GEXP, math_plus<get_value_type_t<GEXP> > >
operator+(GRID_EXPR(GEXP) const& gexp, get_value_type_t<GEXP> const& value)
{
    typedef math_plus<get_value_type_t<GEXP> > bin_op_type;
    return binary_scalar_grid_expression1<GEXP, bin_op_type>(gexp, bin_op_type(), value);
}

template<class GEXP>
binary_scalar_grid_expression1<GEXP, math_minus<get_value_type_t<GEXP> > >
operator-(GRID_EXPR(GEXP) const& gexp, get_value_type_t<GEXP> const& value)
{
    typedef math_minus<get_value_type_t<GEXP> > bin_op_type;
    return binary_scalar_grid_expression1<GEXP, bin_op_type>(gexp, bin_op_type(), value);
}

template<class GEXP>
binary_scalar_grid_expression1<GEXP,
    generic_multiplies<
        get_value_type_t<GEXP>,
        get_scalar_type_t<get_value_type_t<GEXP> >,
        get_value_type_t<GEXP>
    >
> operator*(GRID_EXPR(GEXP) const& gexp, get_scalar_type_t<get_value_type_t<GEXP> > const& value)
{
    typedef generic_multiplies<
        get_value_type_t<GEXP>,
        get_scalar_type_t<get_value_type_t<GEXP> >,
        get_value_type_t<GEXP>
    > bin_op_type;
    return binary_scalar_grid_expression1<GEXP, bin_op_type>(gexp, bin_op_type(), value);
}

template<class GEXP>
binary_scalar_grid_expression1<GEXP,
    generic_divides<
        get_value_type_t<GEXP>,
        get_scalar_type_t<get_value_type_t<GEXP> >,
        get_value_type_t<GEXP>
    >
> operator/(GRID_EXPR(GEXP) const& gexp, get_scalar_type_t<get_value_type_t<GEXP> > const& value)
{
    typedef generic_divides<
        get_value_type_t<GEXP>,
        get_scalar_type_t<get_value_type_t<GEXP> >,
        get_value_type_t<GEXP>
    > bin_op_type;
    return binary_scalar_grid_expression1<GEXP, bin_op_type>(gexp, bin_op_type(), value);
}

_KIAM_MATH_END
