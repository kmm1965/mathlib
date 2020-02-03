#pragma once

#include "iterator_support.hpp"

_KIAM_MATH_BEGIN

template<typename IT>
struct matrix_stride_iterator : std::iterator<
    typename std::iterator_traits<IT>::iterator_category,
    typename std::iterator_traits<IT>::value_type,
    math_pair<
        typename std::iterator_traits<IT>::difference_type,
        typename std::iterator_traits<IT>::difference_type
    >,
    void,
    typename std::iterator_traits<IT>::reference
    
>{
    typedef IT inner_iterator_type;
    MATH_STATIC_ASSERT((std::is_convertible<typename std::iterator_traits<inner_iterator_type>::iterator_category, std::random_access_iterator_tag>::value));

    typedef std::iterator<
        typename std::iterator_traits<inner_iterator_type>::iterator_category,
        typename std::iterator_traits<inner_iterator_type>::value_type,
        math_pair<
            typename std::iterator_traits<inner_iterator_type>::difference_type,
            typename std::iterator_traits<inner_iterator_type>::difference_type
        >,
        void,
        typename std::iterator_traits<inner_iterator_type>::reference
    > super;
    typedef typename super::value_type value_type;
    typedef typename super::difference_type difference_type;
    typedef typename super::pointer pointer;
    typedef typename super::const_pointer const_pointer;
    typedef typename super::reference reference;
    typedef typename super::const_reference const_reference;

    matrix_stride_iterator(inner_iterator_type inner_iterator, unsigned stride_x, unsigned stride_y) :
        m_inner_iterator(inner_iterator), m_stride_x(stride_x), m_stride_y(stride_y){}

    matrix_stride_iterator(const matrix_stride_iterator &other) :
        m_inner_iterator(other.inner_iterator()), m_stride_x(other.stride_x()), m_stride_y(other.stride_y()){}

    inner_iterator_type inner_iterator() const { return m_inner_iterator; }

    unsigned stride_x() const { return m_stride_x; }

    unsigned stride_y() const { return m_stride_y; }

    reference operator*() const {
        return *m_inner_iterator;
    }

    matrix_stride_iterator operator+(const difference_type &d) const {
        return matrix_stride_iterator(m_inner_iterator + (d.first * m_stride_x + d.second * m_stride_y), m_stride_x, m_stride_y);
    }

    matrix_stride_iterator operator-(const difference_type &d) const {
        return matrix_stride_iterator(m_inner_iterator - (d.first * m_stride_x + d.second * m_stride_y), m_stride_x, m_stride_y);
    }

    void operator+=(const difference_type &d){
        m_inner_iterator += (d.first * m_stride_x + d.second * m_stride_y);
    }

    void operator-=(const difference_type &d){
        m_inner_iterator -= (d.first * m_stride_x + d.second * m_stride_y);
    }

    reference operator[](const difference_type &d) const {
        return *(*this + d);
    }

    reference operator()(unsigned i, unsigned j) const {
        return (*this)[difference_type(i, j)];
    }

    bool operator==(const matrix_stride_iterator &other) const {
        return m_inner_iterator == other.inner_iterator() && m_stride_x == other.stride_x() && m_stride_y == other.stride_y();
    }

    bool operator!=(const matrix_stride_iterator &other) const {
        return m_inner_iterator != other.inner_iterator() || m_stride_x != other.stride_x() || m_stride_y != other.stride_y();
    }

    difference_type operator-(const matrix_stride_iterator &other) const
    {
        assert(m_stride_x == other.stride_x());
        assert(m_stride_y == other.stride_y());
        const difference_type diff = m_inner_iterator - other.inner_iterator();
        return m_stride_x > m_stride_y ?
            difference_type(diff / m_stride_x, diff % m_stride_x / m_stride_y) :
            difference_type(diff / m_stride_y, diff % m_stride_y / m_stride_x);
    }

private:
    inner_iterator_type m_inner_iterator;
    const unsigned m_stride_x, m_stride_y;
};

template<typename IT>
matrix_stride_iterator<IT> get_matrix_stride_iterator(IT iterator, unsigned stride_x, unsigned stride_y)
{
    return matrix_stride_iterator<IT>(iterator, stride_x, stride_y);
}

template<typename T>
struct proxy_matrix_stride_iterator : std::iterator<std::random_access_iterator_tag, T>
{
    typedef std::iterator<std::random_access_iterator_tag, T> super;
    typedef typename super::value_type value_type;
    typedef typename super::pointer pointer;
    typedef typename super::const_pointer const_pointer;
    typedef typename super::reference reference;
    typedef typename super::const_reference const_reference;

    template<typename IT>
    proxy_matrix_stride_iterator(matrix_stride_iterator<IT> &parent) :
        m_data(get_iterator_data_pointer(parent)), m_cdata(m_data), m_stride_x(parent.stride_x()), m_stride_y(parent.stride_y())
    {
        MATH_STATIC_ASSERT((std::is_same<value_type, typename std::iterator_traits<IT>::value_type>::value));
    }

    template<typename IT>
    proxy_matrix_stride_iterator(const matrix_stride_iterator<IT> &parent) :
        m_data(0), m_cdata(get_iterator_data_pointer(parent)), m_stride_x(parent.stride_x()), m_stride_y(parent.stride_y())
    {
        MATH_STATIC_ASSERT((std::is_same<value_type, typename std::iterator_traits<IT>::value_type>::value));
    }

    __DEVICE __HOST
    CONSTEXPR unsigned stride_x() const { return m_stride_x; }

    __DEVICE __HOST
    CONSTEXPR unsigned stride_y() const { return m_stride_y; }

    __DEVICE __HOST
    CONSTEXPR reference operator()(unsigned i, unsigned j)
    {
#ifndef __CUDACC__
        assert(m_data != 0);
#endif
        return m_data[m_stride_x * i + m_stride_y * j];
    }

    __DEVICE __HOST
    CONSTEXPR const_reference operator()(unsigned i, unsigned j) const {
        return m_cdata[m_stride_x * i + m_stride_y * j];
    }

private:
    pointer m_data;
    const_pointer m_cdata;
    const unsigned m_stride_x, m_stride_y;
};

_KIAM_MATH_END
