#pragma once

#include "iterator_support.hpp"

_KIAM_MATH_BEGIN

template<typename IT>
struct stride_iterator : std::iterator<
	typename std::iterator_traits<IT>::iterator_category,
	typename std::iterator_traits<IT>::value_type,
	typename std::iterator_traits<IT>::difference_type,
	void,
	typename std::iterator_traits<IT>::reference
>{
	typedef IT inner_iterator_type;
	typedef std::iterator<
		typename std::iterator_traits<inner_iterator_type>::iterator_category,
		typename std::iterator_traits<inner_iterator_type>::value_type,
		typename std::iterator_traits<inner_iterator_type>::difference_type,
		void,
		typename std::iterator_traits<inner_iterator_type>::reference
	> super;
	typedef typename super::value_type value_type;
	typedef typename super::difference_type difference_type;
	typedef typename super::pointer pointer;
	typedef typename super::const_pointer const_pointer;
	typedef typename super::reference reference;
	typedef typename super::const_reference const_reference;

	__DEVICE __HOST
	CONSTEXPR stride_iterator(inner_iterator_type inner_iterator, int stride) : m_inner_iterator(inner_iterator), m_stride(stride){}

	__DEVICE __HOST
    CONSTEXPR stride_iterator(const stride_iterator &other) : m_inner_iterator(other.inner_iterator()), m_stride(other.stride()){}

	__DEVICE __HOST
    CONSTEXPR stride_iterator& operator=(const stride_iterator &other)
	{
		if (this != &other){
			m_inner_iterator = other.inner_iterator();
			m_stride = other.stride();
		}
		return *this;
	}

	__DEVICE __HOST
    CONSTEXPR inner_iterator_type inner_iterator() const { return m_inner_iterator; }
	
	__DEVICE __HOST
    CONSTEXPR int stride() const { return m_stride; }

    __DEVICE __HOST
    CONSTEXPR reference operator*() const {
		return *m_inner_iterator;
	}

    __DEVICE __HOST
    CONSTEXPR stride_iterator operator++(){
		m_inner_iterator += m_stride;
		return *this;
	}

	__DEVICE __HOST
    CONSTEXPR stride_iterator operator++(int)
	{
		stride_iterator result = *this;
		++(*this);
		return result;
	}

    __DEVICE __HOST
    CONSTEXPR stride_iterator operator--(){
		m_inner_iterator -= m_stride;
		return *this;
	}

    __DEVICE __HOST
    CONSTEXPR stride_iterator operator--(int)
	{
		stride_iterator result = *this;
		--(*this);
		return result;
	}

    __DEVICE __HOST
    CONSTEXPR stride_iterator operator+(difference_type n) const {
		return stride_iterator(m_inner_iterator + n * m_stride, m_stride);
	}

    __DEVICE __HOST
    CONSTEXPR stride_iterator operator-(difference_type n) const {
		return stride_iterator(m_inner_iterator - n * m_stride, m_stride);
	}

    __DEVICE __HOST
    CONSTEXPR void operator+=(difference_type n){
		m_inner_iterator += n * m_stride;
	}

    __DEVICE __HOST
    CONSTEXPR void operator-=(difference_type n){
		m_inner_iterator -= n * m_stride;
	}

    __DEVICE __HOST
    CONSTEXPR reference operator[](difference_type n) const {
		return *(*this + n);
	}

    __DEVICE __HOST
    CONSTEXPR bool operator==(const stride_iterator &other) const {
		return m_inner_iterator == other.inner_iterator() && m_stride == other.stride();
	}

    __DEVICE __HOST
    CONSTEXPR bool operator!=(const stride_iterator &other) const {
		return m_inner_iterator != other.inner_iterator() || m_stride != other.stride();
	}

    __DEVICE __HOST
    CONSTEXPR bool operator<(const stride_iterator &other) const
	{
		assert(m_stride == other.stride());
		return m_inner_iterator < other.inner_iterator();
	}

    __DEVICE __HOST
    CONSTEXPR difference_type operator-(const stride_iterator &other) const
	{
		assert(m_stride == other.stride());
		return (m_inner_iterator - other.inner_iterator()) / m_stride;
	}

private:
	inner_iterator_type m_inner_iterator;
	int m_stride;
};

template<typename IT>
__DEVICE __HOST
CONSTEXPR stride_iterator<IT> get_stride_iterator(IT iterator, int stride){
	return stride_iterator<IT>(iterator, stride);
}

template<typename T>
struct proxy_stride_iterator : std::iterator<std::random_access_iterator_tag, T>
{
	typedef std::iterator<std::random_access_iterator_tag, T> super;
	typedef typename super::value_type value_type;
	typedef typename super::pointer pointer;
	typedef typename super::const_pointer const_pointer;
	typedef typename super::reference reference;
	typedef typename super::const_reference const_reference;

	template<typename IT>
	proxy_stride_iterator(IT &parent) : m_data(get_iterator_data_pointer(parent)), m_cdata(m_data), m_stride(get_iterator_stride(parent)){
		MATH_STATIC_ASSERT((std::is_same<value_type, typename std::iterator_traits<IT>::value_type>::value));
	}

	template<typename IT>
	proxy_stride_iterator(const IT &parent) : m_data(0), m_cdata(get_iterator_data_pointer(parent)), m_stride(get_iterator_stride(parent)){
		MATH_STATIC_ASSERT((std::is_same<value_type, typename std::iterator_traits<IT>::value_type>::value));
	}

	__DEVICE __HOST
	CONSTEXPR pointer data()
	{
#ifndef __CUDACC__
		assert(m_data != 0);
#endif
		return m_data;
	}
	
	__DEVICE __HOST
    CONSTEXPR const_pointer data() const { return m_cdata; }

	__DEVICE __HOST
    CONSTEXPR int stride() const { return m_stride; }

	__DEVICE __HOST
    CONSTEXPR reference operator[](int i)
	{
#ifndef __CUDACC__
		assert(m_data != 0);
#endif
		return m_data[m_stride * i];
	}

	__DEVICE __HOST
    CONSTEXPR const_reference operator[](int i) const {
		return m_cdata[m_stride * i];
	}

private:
	pointer m_data;
	const_pointer m_cdata;
	const int m_stride;
};

_KIAM_MATH_END
