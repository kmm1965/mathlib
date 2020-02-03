#pragma once

#include "periodic_index.hpp"

template<typename T, typename TAG>
struct first_derivative_operator;

template<typename T>
struct first_derivative_operator<T, tag_main> : periodic_index_operator<tag_main, first_derivative_operator<T, tag_main> >
{
    typedef periodic_index_operator<tag_main, first_derivative_operator> super;
    typedef T value_type;

    constexpr first_derivative_operator(typename super::index_type const& index, T h) : super(index), h(h){};

    template<typename EOP>
    constexpr get_value_type_t<EOP> operator()(size_t i, const EOP& eobj_proxy) const {
        return (eobj_proxy[super::m_index[super::m_index.value(i) + 1]] - eobj_proxy[i]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator)

private:
    const value_type h;
};

template<typename T>
struct first_derivative_operator<T, tag_aux> : periodic_index_operator<tag_aux, first_derivative_operator<T, tag_aux> >
{
	typedef periodic_index_operator<tag_aux, first_derivative_operator> super;
    typedef T value_type;

    constexpr first_derivative_operator(typename super::index_type const& index, T h) : super(index), h(h){};

    template<typename EOP>
    constexpr get_value_type_t<EOP> operator()(size_t i, const EOP& eobj_proxy) const {
        return (eobj_proxy[i] - eobj_proxy[super::m_index[super::m_index.value(i) - 1]]) / h;
    }

    IMPLEMENT_MATH_EVAL_OPERATOR(first_derivative_operator)

private:
    const value_type h;
};
