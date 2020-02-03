#pragma once

#include "periodic_index.hpp"

template<typename TAG>
struct average_operator;

template<>
struct average_operator<tag_main> : periodic_index_operator<tag_main, average_operator<tag_main> >
{
    typedef periodic_index_operator<tag_main, average_operator> super;

	constexpr average_operator(typename super::index_type const& index) : super(index) {}

	template<typename EOP>
	constexpr get_value_type_t<EOP> operator()(size_t i, EOP const& eobj_proxy) const {
		return 0.5 * (eobj_proxy[i] + eobj_proxy[super::m_index[super::m_index.value(i) + 1]]);
	}
	
	IMPLEMENT_MATH_EVAL_OPERATOR(average_operator)
};

template<>
struct average_operator<tag_aux> : periodic_index_operator<tag_aux, average_operator<tag_aux> >
{
    typedef periodic_index_operator<tag_aux, average_operator> super;

	constexpr average_operator(typename super::index_type const& index) : super(index) {}

	template<typename EOP>
	constexpr get_value_type_t<EOP> operator()(size_t i, EOP const& eobj_proxy) const {
		return 0.5 * (eobj_proxy[super::m_index[super::m_index.value(i) - 1]] + eobj_proxy[i]);
	}
	
	IMPLEMENT_MATH_EVAL_OPERATOR(average_operator)
};
