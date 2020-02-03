#pragma once

#include "math_object.hpp"

_KIAM_MATH_BEGIN

template<class INDEX, class _Proxy = INDEX>
struct index_base : math_object<INDEX, _Proxy>
{
protected:
    CONSTEXPR index_base(){} // protect from direct usage
};

struct default_index : index_base<default_index>
{
    typedef size_t value_type;

    CONSTEXPR default_index(size_t size) : m_size(size) {}

    __DEVICE
    CONSTEXPR size_t size() const {
        return m_size;
    }

    __DEVICE
    CONSTEXPR size_t operator[](value_type i) const
    {
        assert(i < m_size);
        return i;
    }

    __DEVICE
    CONSTEXPR value_type value(size_t i) const
    {
        assert(i < m_size);
        return i;
    }

private:
    const size_t m_size;
};

_KIAM_MATH_END
