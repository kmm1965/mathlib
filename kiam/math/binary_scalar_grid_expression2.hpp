#pragma once

#include "grid_expression.hpp"
#include "context.hpp"

_KIAM_MATH_BEGIN

template<class GEXP, class BO>
struct binary_scalar_grid_expression2 : grid_expression<typename GEXP::tag_type, binary_scalar_grid_expression2<GEXP, BO> >
{
    typedef BO bin_op_type;
    typedef typename bin_op_type::first_argument_type arg_type;
    typedef typename bin_op_type::result_type value_type;

    binary_scalar_grid_expression2(arg_type const& value, bin_op_type const& bin_op, GRID_EXPR(GEXP) const& gexp) :
        value(value), bin_op(bin_op), gexp_proxy(gexp.get_proxy()){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return bin_op(value, gexp_proxy[i]);
    }

    __DEVICE
    CONSTEXPR value_type operator()(size_t i) const {
        return bin_op(value, gexp_proxy(i));
    }

    template<typename CONTEXT>
    __DEVICE
    CONSTEXPR value_type operator()(size_t i, context<CONTEXT> const& context) const {
        return bin_op(value, gexp_proxy(i, context));
    }

private:
    arg_type const value;
    bin_op_type const bin_op;
    typename GEXP::proxy_type const gexp_proxy;
};

template<class GEXP>
binary_scalar_grid_expression2<GEXP, math_plus<get_value_type_t<GEXP> > >
operator+(get_value_type_t<GEXP> const& value, GRID_EXPR(GEXP) const& gexp)
{
    typedef math_plus<get_value_type_t<GEXP> > bin_op_type;
    return binary_scalar_grid_expression2<GEXP, bin_op_type>(value, bin_op_type(), gexp);
}

template<class GEXP>
binary_scalar_grid_expression2<GEXP, math_minus<get_value_type_t<GEXP> > >
operator-(get_value_type_t<GEXP> const& value, GRID_EXPR(GEXP) const& gexp)
{
    typedef math_minus<get_value_type_t<GEXP> > bin_op_type;
    return binary_scalar_grid_expression2<GEXP, bin_op_type>(value, bin_op_type(), gexp);
}

template<class GEXP>
binary_scalar_grid_expression2<GEXP,
    generic_multiplies<
        get_scalar_type_t<get_value_type_t<GEXP> >,
        get_value_type_t<GEXP>,
        get_value_type_t<GEXP>
    >
> operator*(get_scalar_type_t<get_value_type_t<GEXP> > const& value, GRID_EXPR(GEXP) const& gexp)
{
    typedef generic_multiplies<
        get_scalar_type_t<get_value_type_t<GEXP> >,
        get_value_type_t<GEXP>,
        get_value_type_t<GEXP>
    > bin_op_type;
    return binary_scalar_grid_expression2<GEXP, bin_op_type>(value, bin_op_type(), gexp);
}

template<class GEXP>
typename std::enable_if<
    supports_divides<get_scalar_type_t<get_value_type_t<GEXP> >, get_value_type_t<GEXP> >::value,
    binary_scalar_grid_expression2<GEXP,
        generic_divides<
            get_scalar_type_t<get_value_type_t<GEXP> >,
            get_value_type_t<GEXP>,
            get_value_type_t<GEXP>
        >
    >
>::type operator/(get_scalar_type_t<get_value_type_t<GEXP> > const& value, GRID_EXPR(GEXP) const& gexp)
{
    typedef generic_divides<
        get_scalar_type_t<get_value_type_t<GEXP> >,
        get_value_type_t<GEXP>,
        get_value_type_t<GEXP>
    > bin_op_type;
    return binary_scalar_grid_expression2<GEXP, bin_op_type>(value, bin_op_type(), gexp);
}

_KIAM_MATH_END
