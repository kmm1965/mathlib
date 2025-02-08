#pragma once

#include "math_def.h"

#include <boost/units/quantity.hpp>
#include <boost/units/is_quantity.hpp>

_KIAM_MATH_BEGIN

template<class Unit, class Y>
struct get_scalar_type<boost::units::quantity<Unit, Y> >
{
    typedef Y type;
};

template<class Unit1, class Unit2, class Y>
struct supports_multiplies<boost::units::quantity<Unit1, Y>, boost::units::quantity<Unit2, Y> > : std::is_same<typename Unit1::system_type, typename Unit2::system_type> {};

template<class Unit1, class Unit2, class Y>
struct multiplies_result_type<boost::units::quantity<Unit1, Y>, boost::units::quantity<Unit2, Y> >
{
    static_assert(std::is_same<typename Unit1::system_type, typename Unit2::system_type>::value, "System types should be the same");
    typedef boost::units::quantity<typename boost::units::multiply_typeof_helper<Unit1, Unit2>::type, Y> type;
};

template<class Unit1, class Unit2, class Y>
struct supports_divides<boost::units::quantity<Unit1, Y>, boost::units::quantity<Unit2, Y> > : std::is_same<typename Unit1::system_type, typename Unit2::system_type> {};

template<class Unit1, class Unit2, class Y>
struct divides_result_type<boost::units::quantity<Unit1, Y>, boost::units::quantity<Unit2, Y> >
{
    static_assert(std::is_same<typename Unit1::system_type, typename Unit2::system_type>::value, "System types should be the same");
    typedef boost::units::quantity<typename boost::units::divide_typeof_helper<Unit1, Unit2>::type, Y> type;
};

template<class Unit, class Y>
struct supports_divides<Y, boost::units::quantity<Unit, Y> > : std::true_type {};

template<class Unit, class Y>
struct divides_result_type<Y, boost::units::quantity<Unit, Y> >
{
    typedef boost::units::quantity<
        typename boost::units::divide_typeof_helper<
            boost::units::unit<boost::units::dimensionless_type, typename Unit::system_type>,
            Unit
        >::type, Y
    > type;
};

template<class GEXP, class Unit, class Y>
typename std::enable_if<
    boost::units::is_quantity<get_value_type_t<GEXP> >::value,
    binary_scalar_grid_expression1<GEXP,
        generic_multiplies<
            get_value_type_t<GEXP>,
            boost::units::quantity<Unit, Y>,
            typename boost::units::multiply_typeof_helper<get_value_type_t<GEXP>, boost::units::quantity<Unit, Y> >::type
        >
    >
>::type operator*(GRID_EXPR(GEXP) const& gexp, boost::units::quantity<Unit, Y> const& value)
{
    typedef generic_multiplies<
        get_value_type_t<GEXP>,
        boost::units::quantity<Unit, Y>,
        typename boost::units::multiply_typeof_helper<get_value_type_t<GEXP>, boost::units::quantity<Unit, Y> >::type
    > bin_op_type;
    return binary_scalar_grid_expression1<GEXP, bin_op_type>(gexp, bin_op_type(), value);
}

template<class GEXP, class Unit, class Y>
typename std::enable_if<
    boost::units::is_quantity<get_value_type_t<GEXP> >::value,
    binary_scalar_grid_expression1<GEXP,
        generic_divides<
            get_value_type_t<GEXP>,
            boost::units::quantity<Unit, Y>,
            typename boost::units::divide_typeof_helper<get_value_type_t<GEXP>, boost::units::quantity<Unit, Y> >::type
        >
    >
>::type operator/(GRID_EXPR(GEXP) const& gexp, boost::units::quantity<Unit, Y> const& value)
{
    typedef generic_divides<
        get_value_type_t<GEXP>,
        boost::units::quantity<Unit, Y>,
        typename boost::units::divide_typeof_helper<get_value_type_t<GEXP>, boost::units::quantity<Unit, Y> >::type
    > bin_op_type;
    return binary_scalar_grid_expression1<GEXP, bin_op_type>(gexp, bin_op_type(), value);
}

template<class GEXP, class Unit, class Y>
typename std::enable_if<
    boost::units::is_quantity<get_value_type_t<GEXP> >::value,
    binary_scalar_grid_expression2<GEXP,
        generic_multiplies<
            boost::units::quantity<Unit, Y>,
            get_value_type_t<GEXP>,
            typename boost::units::multiply_typeof_helper<boost::units::quantity<Unit, Y>, get_value_type_t<GEXP> >::type
        >
    >
>::type operator*(boost::units::quantity<Unit, Y> const& value, GRID_EXPR(GEXP) const& gexp)
{
    typedef generic_multiplies<
        boost::units::quantity<Unit, Y>,
        get_value_type_t<GEXP>,
        typename boost::units::multiply_typeof_helper<get_value_type_t<GEXP>, boost::units::quantity<Unit, Y> >::type
    > bin_op_type;
    return binary_scalar_grid_expression2<GEXP, bin_op_type>(value, bin_op_type(), gexp);
}

template<class Unit, class Y>
struct is_dimensionless<boost::units::quantity<Unit, Y> > : std::false_type {};

template<class System, class Y>
struct is_dimensionless<boost::units::quantity<boost::units::unit<boost::units::dimensionless_type, System>, Y> > : std::true_type {};

template<class Unit, class Y>
struct sqrt_result_type<boost::units::quantity<Unit, Y> >
{
    typedef typename boost::units::root_typeof_helper<boost::units::quantity<Unit, Y>, boost::units::static_rational<2> >::type type;
};

template<class Unit, class Y>
struct sqr_result_type<boost::units::quantity<Unit, Y> >
{
    typedef boost::units::quantity<typename boost::units::multiply_typeof_helper<Unit, Unit>::type, Y> type;
};

_KIAM_MATH_END
