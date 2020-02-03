#pragma once

#include "../gridop2_src/periodic_index.hpp"

template<class BASE>
struct first_derivative_operator_base : BASE
{
    typedef BASE super;

    template<typename T>
    struct get_value_type;

    template<class Unit, class Y>
    struct get_value_type<boost::units::quantity<Unit, Y> >
    {
        typedef typename boost::units::divide_typeof_helper<boost::units::quantity<Unit, Y>, length_type>::type type;
    };

protected:
    first_derivative_operator_base(typename super::index_type const& index) : super(index){}
};

template<typename TAG_X>
struct first_derivative_operator_x;

template<>
struct first_derivative_operator_x<tag_main> : first_derivative_operator_base<periodic_index_operator_x<tag_main, first_derivative_operator_x<tag_main> > >
{
	typedef first_derivative_operator_base<periodic_index_operator_x<tag_main, first_derivative_operator_x> > super;

    first_derivative_operator_x(typename super::index_type const& index, const length_type &h) : super(index), h(h){};

    template<typename EOP>
    __DEVICE
    constexpr typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return (eobj_proxy[super::m_index[typename super::index_value_type(ind.first + 1, ind.second)]] - eobj_proxy[i]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_x)

private:
    const length_type h;
};

template<>
struct first_derivative_operator_x<tag_aux> : first_derivative_operator_base<periodic_index_operator_x<tag_aux, first_derivative_operator_x<tag_aux> > >
{
	typedef first_derivative_operator_base<periodic_index_operator_x<tag_aux, first_derivative_operator_x> > super;

    first_derivative_operator_x(typename super::index_type const& index, const length_type &h) : super(index), h(h){};

    template<typename EOP>
    __DEVICE
    constexpr typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return (eobj_proxy[i] - eobj_proxy[super::m_index[typename super::index_value_type(ind.first - 1, ind.second)]]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_x)

private:
    const length_type h;
};

/////////////
template<typename TAG_Y>
struct first_derivative_operator_y;

template<>
struct first_derivative_operator_y<tag_main> : first_derivative_operator_base<periodic_index_operator_y<tag_main, first_derivative_operator_y<tag_main> > >
{
	typedef first_derivative_operator_base<periodic_index_operator_y<tag_main, first_derivative_operator_y> > super;

    first_derivative_operator_y(typename super::index_type const& index, const length_type &h) : super(index), h(h){};

    template<typename EOP>
    __DEVICE
    constexpr typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return (eobj_proxy[super::m_index[typename super::index_value_type(ind.first, ind.second + 1)]] - eobj_proxy[i]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_y)

private:
    const length_type h;
};

template<>
struct first_derivative_operator_y<tag_aux> : first_derivative_operator_base<periodic_index_operator_y<tag_aux, first_derivative_operator_y<tag_aux> > >
{
	typedef first_derivative_operator_base<periodic_index_operator_y<tag_aux, first_derivative_operator_y> > super;

    first_derivative_operator_y(typename super::index_type const& index, const length_type &h) : super(index), h(h){};

    template<typename EOP>
    __DEVICE
    constexpr typename super::template get_value_type<get_value_type_t<EOP> >::type operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return (eobj_proxy[i] - eobj_proxy[super::m_index[typename super::index_value_type(ind.first, ind.second - 1)]]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_y)

private:
    const length_type h;
};
