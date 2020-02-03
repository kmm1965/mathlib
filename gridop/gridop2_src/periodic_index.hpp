#pragma once

#include "cyclic.hpp"

struct periodic_index : index_base<periodic_index>
{
    typedef isize_t value_type;

    periodic_index(size_t size) : m_size(size){}

    constexpr size_t size() const {
        return m_size;
    }

    constexpr size_t operator[](value_type i) const
    {
        const value_type N = (value_type)m_size;
        i %= N;
        assert(i > -N && i < N);
        if (i < 0)
            i += N;
        assert(i >= 0 && i < N);
        return i;
    }

    constexpr value_type value(size_t i) const
    {
        assert(i < m_size);
        return i;
    }

private:
    const size_t m_size;
};

template<class MO>
struct periodic_index_operator : math_operator<MO>
{
    typedef dim2_index<periodic_index, periodic_index> index_type;
    typedef get_value_type_t<index_type> index_value_type;
    typedef get_value_type_t<periodic_index> periodic_index_value_type;

protected:
    periodic_index_operator(index_type const& index) : m_index(index.get_proxy()){}

public:
    constexpr typename index_type::proxy_type const& index() const {
        return m_index;
    }

protected:
    const typename index_type::proxy_type m_index;
};

template<class TAG, class MO>
struct periodic_index_operator_x : periodic_index_operator<MO>
{
    typedef periodic_index_operator<MO> super;

    template<class EO_TAG>
    struct get_tag_type;

    template<typename tag_y>
    struct get_tag_type<std::tuple<TAG, tag_y> >
    {
        typedef std::tuple<other_tag_t<TAG>, tag_y> type;
    };

protected:
    periodic_index_operator_x(typename super::index_type const& index) : super(index){}
};

template<class TAG, class MO>
struct periodic_index_operator_y : periodic_index_operator<MO>
{
    typedef periodic_index_operator<MO> super;

    template<class EO_TAG>
    struct get_tag_type;

    template<typename tag_x>
    struct get_tag_type<std::tuple<tag_x, TAG> >
    {
        typedef std::tuple<tag_x, other_tag_t<TAG> > type;
    };

protected:
    periodic_index_operator_y(typename super::index_type const& index) : super(index) {}
};
