#pragma once

#include "index_base.hpp"

_KIAM_MATH_BEGIN

template<typename IR, typename IC>
struct dim2_index : index_base<dim2_index<IR, IC> >
{
    typedef math_pair<get_value_type_t<IR>, get_value_type_t<IC> > value_type;

    dim2_index(IR const& row_index, IC const& column_index) : m_row_index(row_index.get_proxy()), m_column_index(column_index.get_proxy()){}

    __DEVICE
    CONSTEXPR typename IR::proxy_type const& row_index() const {
        return m_row_index;
    }

    __DEVICE
    CONSTEXPR typename IC::proxy_type const& column_index() const {
        return m_column_index;
    }

    __DEVICE
    CONSTEXPR size_t operator[](value_type const& i) const {
        return m_column_index[i.second] * m_row_index.size() + m_row_index[i.first];
    }

    __DEVICE
    CONSTEXPR value_type value(size_t i) const {
        return value_type(m_row_index.value(i % m_row_index.size()), m_column_index.value(i / m_row_index.size()));
    }

private:
    const typename IR::proxy_type m_row_index;
    const typename IC::proxy_type m_column_index;
};

_KIAM_MATH_END
