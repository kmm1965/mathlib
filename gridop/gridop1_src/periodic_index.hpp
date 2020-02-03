#pragma once

#include "cyclic.hpp"

struct periodic_index : index_base<periodic_index>
{
	typedef isize_t value_type;

    constexpr periodic_index(size_t size) : m_size(size){}

    constexpr size_t size() const {
        return m_size;
    }

    constexpr size_t operator[](value_type i) const
	{
        const value_type N = (value_type)m_size;
        return (i + N) % N;
	}

    constexpr value_type value(size_t i) const
    {
        assert(i < m_size);
        return i;
    }

private:
    const size_t m_size;
};

template<class TAG, class MO>
struct periodic_index_operator : math_operator<MO>
{
    typedef periodic_index index_type;
    typedef get_value_type_t<index_type> index_value_type;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4348) // redefinition of default parameter: parameter 2
#endif
    // Cannot make explicit specialization in non-namespace scope
    template<class EO_TAG, typename unused = void>
    struct get_tag_type;

    template<typename unused>
    struct get_tag_type<TAG, unused>
    {
        typedef other_tag_t<TAG> type;
    };

#ifdef _MSC_VER
#pragma warning(pop)
#endif

protected:
    constexpr periodic_index_operator(index_type const& index) : m_index(index.get_proxy()) {}

public:
    constexpr typename index_type::proxy_type const& index() const {
        return m_index;
    }

protected:
    const typename index_type::proxy_type m_index;
};
