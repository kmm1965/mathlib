#pragma once

#include "periodic_index.hpp"

template<typename TAG_X>
struct average_operator_x;

template<>
struct average_operator_x<tag_main> : periodic_index_operator_x<tag_main, average_operator_x<tag_main> >
{
    typedef periodic_index_operator_x<tag_main, average_operator_x> super;

    average_operator_x(typename super::index_type const& index) : super(index) {}

    template<typename EOP>
    __DEVICE
    constexpr get_value_type_t<EOP> operator()(size_t i, EOP const& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return 0.5 * (eobj_proxy[i] + eobj_proxy[super::m_index[typename super::index_value_type(ind.first + 1, ind.second)]]);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(average_operator_x)
};

template<>
struct average_operator_x<tag_aux> : periodic_index_operator_x<tag_aux, average_operator_x<tag_aux> >
{
    typedef periodic_index_operator_x<tag_aux, average_operator_x> super;

    average_operator_x(typename super::index_type const& index) : super(index) {}

    template<typename EOP>
    __DEVICE
    constexpr get_value_type_t<EOP> operator()(size_t i, EOP const& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return 0.5 * (eobj_proxy[i] + eobj_proxy[super::m_index[typename super::index_value_type(ind.first - 1, ind.second)]]);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(average_operator_x)
};

template<typename TAG_Y>
struct average_operator_y;

template<>
struct average_operator_y<tag_main> : periodic_index_operator_y<tag_main, average_operator_y<tag_main> >
{
	typedef periodic_index_operator_y<tag_main, average_operator_y> super;

    average_operator_y(typename super::index_type const& index) : super(index) {}

    template<typename EOP>
    __DEVICE
    constexpr get_value_type_t<EOP> operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return 0.5 * (eobj_proxy[i] + eobj_proxy[super::m_index[typename super::index_value_type(ind.first, ind.second + 1)]]);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(average_operator_y)
};

template<>
struct average_operator_y<tag_aux> : periodic_index_operator_y<tag_aux, average_operator_y<tag_aux> >
{
	typedef periodic_index_operator_y<tag_aux, average_operator_y> super;

    average_operator_y(typename super::index_type const& index) : super(index) {}

    template<typename EOP>
    __DEVICE
    constexpr get_value_type_t<EOP> operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return 0.5 * (eobj_proxy[i] + eobj_proxy[super::m_index[typename super::index_value_type(ind.first, ind.second - 1)]]);
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(average_operator_y)
};
