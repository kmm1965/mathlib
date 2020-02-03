#pragma once

#include "periodic_index.hpp"

template<typename T, typename TAG_X>
struct first_derivative_operator_x;

template<typename T>
struct first_derivative_operator_x<T, tag_main> : periodic_index_operator_x<tag_main, first_derivative_operator_x<T, tag_main> >
{
	typedef periodic_index_operator_x<tag_main, first_derivative_operator_x> super;
    typedef T value_type;

    first_derivative_operator_x(typename super::index_type const& index, T h) : super(index), h(h) {};

    template<typename EOP>
    __DEVICE
    constexpr get_value_type_t<EOP> operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return (eobj_proxy[super::m_index[typename super::index_value_type(ind.first + 1, ind.second)]] - eobj_proxy[i]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_x)

private:
    const value_type h;
};

template<typename T>
struct first_derivative_operator_x<T, tag_aux> : periodic_index_operator_x<tag_aux, first_derivative_operator_x<T, tag_aux> >
{
	typedef periodic_index_operator_x<tag_aux, first_derivative_operator_x> super;
    typedef T value_type;

    first_derivative_operator_x(typename super::index_type const& index, T h) : super(index), h(h) {};

    template<typename EOP>
    __DEVICE
    constexpr get_value_type_t<EOP> operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return (eobj_proxy[i] - eobj_proxy[super::m_index[typename super::index_value_type(ind.first - 1, ind.second)]]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_x)

private:
    const value_type h;
};

template<typename T, typename TAG_Y>
struct first_derivative_operator_y;

template<typename T>
struct first_derivative_operator_y<T, tag_main> : periodic_index_operator_y<tag_main, first_derivative_operator_y<T, tag_main> >
{
	typedef periodic_index_operator_y<tag_main, first_derivative_operator_y> super;
    typedef T value_type;

    first_derivative_operator_y(typename super::index_type const& index, T h) : super(index), h(h){};

    template<typename EOP>
    __DEVICE
    constexpr get_value_type_t<EOP> operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return (eobj_proxy[super::m_index[typename super::index_value_type(ind.first, ind.second + 1)]] - eobj_proxy[i]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_y)

private:
    const value_type h;
};

template<typename T>
struct first_derivative_operator_y<T, tag_aux> : periodic_index_operator_y<tag_aux, first_derivative_operator_y<T, tag_aux> >
{
	typedef periodic_index_operator_y<tag_aux, first_derivative_operator_y> super;
    typedef T value_type;

    first_derivative_operator_y(typename super::index_type const& index, T h) : super(index), h(h){};

    template<typename EOP>
    __DEVICE
    constexpr get_value_type_t<EOP> operator()(size_t i, const EOP& eobj_proxy) const
    {
        const typename super::index_value_type ind = super::m_index.value(i);
        return (eobj_proxy[i] - eobj_proxy[super::m_index[typename super::index_value_type(ind.first, ind.second - 1)]]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator_y)

private:
    const value_type h;
};
