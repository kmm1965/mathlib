#pragma once

#include "../gridop1_src/periodic_index.hpp"

template<class TAG, class OP>
struct first_derivative_operator_base : periodic_index_operator<TAG, OP>
{
    typedef periodic_index_operator<TAG, OP> super;

    template<typename T>
    struct get_value_type;

    template<class Unit, class Y>
    struct get_value_type<boost::units::quantity<Unit, Y> >
    {
        typedef typename boost::units::divide_typeof_helper<boost::units::quantity<Unit, Y>, length_type>::type type;
    };

protected:
    constexpr first_derivative_operator_base(typename super::index_type const& index) : super(index) {}
};

template<typename TAG>
struct first_derivative_operator;

template<>
struct first_derivative_operator<tag_main> : first_derivative_operator_base<tag_main, first_derivative_operator<tag_main> >
{
	typedef first_derivative_operator_base<tag_main, first_derivative_operator> super;

    constexpr first_derivative_operator(typename super::index_type const& index, const length_type &h) : super(index), h(h){};

    template<typename EOP>
    constexpr typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i, const EOP& eobj_proxy) const {
        return (eobj_proxy[super::m_index[super::m_index.value(i) + 1]] - eobj_proxy[i]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator)

private:
    const length_type h;
};

template<>
struct first_derivative_operator<tag_aux> : first_derivative_operator_base<tag_aux, first_derivative_operator<tag_aux> >
{
	typedef first_derivative_operator_base<tag_aux, first_derivative_operator> super;

    constexpr first_derivative_operator(typename super::index_type const& index, const length_type &h) : super(index), h(h){};

    template<typename EOP>
    constexpr typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i, const EOP& eobj_proxy) const {
        return (eobj_proxy[i] - eobj_proxy[super::m_index[super::m_index.value(i) - 1]]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator)

private:
    const length_type h;
};
