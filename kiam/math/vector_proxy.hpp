#pragma once

#ifndef __CUDACC__
#include <cassert>
#endif
#include "math_vector.hpp"

_KIAM_MATH_BEGIN

template<typename T>
struct vector_proxy
{
    typedef T value_type;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef pointer iterator;
    typedef const_pointer const_iterator;

    __DEVICE __HOST vector_proxy() : m_size(0), m_data(nullptr), m_cdata(nullptr){}
    __DEVICE __HOST vector_proxy(size_t size, value_type *data) : m_size(size), m_data(data), m_cdata(data){}
    __DEVICE __HOST vector_proxy(size_t size, const value_type *data) : m_size(size), m_data(0), m_cdata(data){}

    vector_proxy(math_vector<value_type> &vec) :
        m_size(vec.size()), m_data(vec.size() == 0 ? nullptr : vec.data_pointer()), m_cdata(vec.size() == 0 ? nullptr : vec.data_pointer()){}

    vector_proxy(math_vector<value_type> const& vec) :
        m_size(vec.size()), m_data(nullptr), m_cdata(vec.size() == 0 ? nullptr : vec.data_pointer()){}

    IMPLEMENT_DEFAULT_COPY_CONSRUCTOR(vector_proxy);

    __DEVICE __HOST
    size_t size() const { return m_size; }

    __DEVICE __HOST
    pointer data()
    {
#ifndef __CUDACC__
        assert(m_data != 0);
#endif
        return m_data;
    }

    __DEVICE __HOST
    const_pointer data() const { return m_cdata; }

    __DEVICE __HOST
    reference operator[](size_t i)
    {
#ifndef __CUDACC__
        assert(i < m_size);
        assert(m_data != 0);
#endif
        return m_data[i];
    }

    __DEVICE __HOST
    const_reference operator[](size_t i) const
    {
#ifndef __CUDACC__
        assert(i < m_size);
#endif
        return m_cdata[i];
    }

    __DEVICE __HOST
    iterator begin()
    {
#ifndef __CUDACC__
        assert(m_data != 0);
#endif
        return m_data;
    }

    __DEVICE __HOST
    iterator end()
    {
#ifndef __CUDACC__
        assert(m_data != 0);
#endif
        return m_data + m_size;
    }

    __DEVICE __HOST
    const_iterator begin() const {
        return m_cdata;
    }

    __DEVICE __HOST
    const_iterator end() const {
        return m_cdata + m_size;
    }

    __DEVICE __HOST
    const_iterator cbegin() const {
        return m_cdata;
    }

    __DEVICE __HOST
    const_iterator cend() const {
        return m_cdata + m_size;
    }

    __DEVICE __HOST
    reference front()
    {
#ifndef __CUDACC__
        assert(m_size > 0);
        assert(m_data != 0);
#endif
        return m_data[0];
    }

    __DEVICE __HOST
    const_reference front() const
    {
#ifndef __CUDACC__
        assert(m_size > 0);
#endif
        return m_cdata[0];
    }

    __DEVICE __HOST
    reference back()
    {
#ifndef __CUDACC__
        assert(m_size > 0);
        assert(m_data != 0);
#endif
        return m_data[m_size - 1];
    }

    __DEVICE __HOST
    const_reference back() const
    {
#ifndef __CUDACC__
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
