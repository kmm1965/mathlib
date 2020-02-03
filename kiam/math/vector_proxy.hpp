#pragma once

#include "math_vector.hpp"

_KIAM_MATH_BEGIN

template<typename T>
struct vector_proxy
{
    typedef T value_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type &reference;
    typedef const value_type &const_reference;
    typedef pointer iterator;
    typedef const_pointer const_iterator;

    CONSTEXPR vector_proxy(size_t size, value_type *data) : m_size(size), m_data(data), m_cdata(data){}
    CONSTEXPR vector_proxy(size_t size, const value_type *data) : m_size(size), m_data(0), m_cdata(data){}

    CONSTEXPR vector_proxy(math_vector<value_type> &vec) :
        m_size(vec.size()), m_data(vec.data_pointer()), m_cdata(vec.data_pointer()){}

    CONSTEXPR vector_proxy(const math_vector<value_type> &vec) :
        m_size(vec.size()), m_data(0), m_cdata(vec.data_pointer()){}

    __DEVICE __HOST
    CONSTEXPR size_t size() const { return m_size; }

    __DEVICE __HOST
    pointer data()
    {
        assert(m_data != 0);
        return m_data;
    }

    __DEVICE __HOST
    CONSTEXPR const_pointer data() const { return m_cdata; }

    __DEVICE __HOST
    reference operator[](size_t i)
    {
        assert(i < m_size);
        assert(m_data != 0);
        return m_data[i];
    }

    __DEVICE __HOST
    CONSTEXPR const_reference operator[](size_t i) const
    {
#ifndef NDEBUG
        assert(i < m_size);
#endif
        return m_cdata[i];
    }

    __DEVICE __HOST
    iterator begin()
    {
        assert(m_data != 0);
        return m_data;
    }

    __DEVICE __HOST
    iterator end()
    {
        assert(m_data != 0);
        return m_data + m_size;
    }

    __DEVICE __HOST
    CONSTEXPR const_iterator begin() const {
        return m_cdata;
    }

    __DEVICE __HOST
    CONSTEXPR const_iterator end() const {
        return m_cdata + m_size;
    }

    __DEVICE __HOST
    CONSTEXPR const_iterator cbegin() const {
        return m_cdata;
    }

    __DEVICE __HOST
    CONSTEXPR const_iterator cend() const {
        return m_cdata + m_size;
    }

    __DEVICE __HOST
    reference front()
    {
        assert(m_size > 0);
        assert(m_data != 0);
        return m_data[0];
    }

    __DEVICE __HOST
    CONSTEXPR const_reference front() const
    {
#ifndef NDEBUG
        assert(m_size > 0);
#endif
        return m_cdata[0];
    }

    __DEVICE __HOST
    reference back()
    {
        assert(m_size > 0);
        assert(m_data != 0);
        return m_data[m_size - 1];
    }

    __DEVICE __HOST
    CONSTEXPR const_reference back() const
    {
#ifndef NDEBUG
        assert(m_size > 0);
#endif
        return m_cdata[m_size - 1];
    }

private:
    size_t const m_size;
    pointer const m_data;
    const_pointer const m_cdata;
};

_KIAM_MATH_END
