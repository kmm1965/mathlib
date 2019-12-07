#pragma once

#include "math_def.h"

#ifndef __CUDACC__

#ifdef DONT_USE_CXX_11
#include <boost/array.hpp>
#else
#include <array>
#endif

#endif	// __CUDACC__

_KIAM_MATH_BEGIN

#ifdef __CUDACC__

template<typename T, size_t N>
struct math_array
{
	typedef T value_type;
	typedef value_type &reference;
	typedef const value_type &const_reference;
	typedef T *iterator;
	typedef const T *const_iterator;

	__device__ __host__
	size_t size() const {
		return N;
	}

	__device__ __host__
	iterator begin(){
		return m_data;
	}

	__device__ __host__
	const_iterator begin() const {
		return m_data;
	}

	__device__ __host__
	iterator end(){
		return m_data + N;
	}

	__device__ __host__
	const_iterator end() const {
		return m_data + N;
	}

	__device__ __host__
	const_iterator cbegin() const {
		return m_data;
	}

	__device__ __host__
	const_iterator cend() const {
		return m_data + N;
	}

	__device__ __host__
	reference operator[](size_t i)
	{
		assert(i < N);
		return m_data[i];
	}

	__device__ __host__
	const_reference operator[](size_t i) const
	{
		assert(i < N);
		return m_data[i];
	}

private:
	value_type m_data[N == 0 ? 1 : N];
};

#else	// __CUDACC__

template<typename T, size_t N>
#ifdef DONT_USE_CXX_11
struct math_array : boost::array<T, N>{};
#else
using math_array = std::array<T, N>;
#endif

#endif	// __CUDACC__

_KIAM_MATH_END
