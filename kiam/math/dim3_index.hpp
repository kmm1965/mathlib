#pragma once

#include "index_base.hpp"

_KIAM_MATH_BEGIN

template<typename IX, typename IY, typename IZ>
struct dim3_index : index_base<dim3_index<IX, IY, IZ> >
{
	typedef std::tuple<get_value_type_t<IX>, get_value_type_t<IY>, get_value_type_t<IZ> > value_type;

    CONSTEXPR dim3_index(IX const& x_index, IY const& y_index, IZ const& z_index) :
		m_x_index(x_index.get_proxy()),
		m_y_index(y_index.get_proxy()),
		m_z_index(z_index.get_proxy()) {}

	__DEVICE
    CONSTEXPR typename IX::proxy_type const& x_index() const {
		return m_x_index;
	}

	__DEVICE
    CONSTEXPR typename IY::proxy_type const& y_index() const {
		return m_y_index;
	}

	__DEVICE
    CONSTEXPR typename IZ::proxy_type const& z_index() const {
		return m_z_index;
	}

	__DEVICE
    CONSTEXPR size_t operator[](value_type const& i) const {
		return (m_z_index[std::get<2>(i)] * m_y_index.size() + m_y_index[std::get<1>(i)]) *
			m_x_index.size() + m_x_index[std::get<0>(i)];
	}

	__DEVICE
    CONSTEXPR value_type value(size_t i) const {
		return std::make_tuple(
			m_x_index.value(i % m_x_index.size()),
			m_y_index.value(i / m_x_index.size() % m_y_index.size()),
			m_z_index.value(i / m_x_index.size() / m_y_index.size()));
	}

private:
	const typename IX::proxy_type m_x_index;
	const typename IY::proxy_type m_y_index;
	const typename IZ::proxy_type m_z_index;
};

_KIAM_MATH_END
