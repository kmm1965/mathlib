#pragma once

#include "math_def.h"

#if defined(__CUDACC__)
#include <thrust/device_vector.h>
#elif defined(__OPENCL__)
#include <boost/compute/container/vector.hpp>
#else
#include <vector>
#endif

_KIAM_MATH_BEGIN

#if defined(__CUDACC__)
#define MATH_VECTOR_BASE_CLASS thrust::device_vector
#define HOST_VECTOR_T _KIAM_MATH::host_vector
#elif defined(__OPENCL__)
#define MATH_VECTOR_BASE_CLASS ::boost::compute::vector
#define HOST_VECTOR_T _KIAM_MATH::host_vector
#else
#define MATH_VECTOR_BASE_CLASS std::vector
#define HOST_VECTOR_T _KIAM_MATH::math_vector
#endif

template<typename T>
struct host_vector;

template<typename T>
struct vector_proxy;

template<typename T>
struct math_vector : public MATH_VECTOR_BASE_CLASS<T>
{
	typedef math_vector type;
	typedef T value_type;
	typedef MATH_VECTOR_BASE_CLASS<value_type> super;
	typedef value_type *pointer;
	typedef const value_type *const_pointer;
	typedef vector_proxy<value_type> vector_proxy_type;

	math_vector() : super(){}
	math_vector(size_t size) : super(size){}
	math_vector(size_t size, const value_type &initValue) : super(size, initValue){}
#ifndef DONT_USE_CXX_11
	math_vector(const math_vector &other) : super(other){};
	math_vector(math_vector &&other) : super(std::forward<super>(other)){}
#endif
	math_vector(const host_vector<value_type>& hv)
#ifdef __CUDACC__
	{ operator=(hv); }
#else
		: super(hv){}
#endif

	void operator=(const math_vector &other){ super::operator=(other); }
	void operator=(math_vector &&other){ super::operator=(std::forward<super>(other)); }

	void operator=(const host_vector<value_type> &hv)
	{
#ifdef __CUDACC__
		super::resize(hv.size());
		thrust::cuda_cub::throw_on_error(cudaMemcpy(data_pointer(), hv.data_pointer(), super::size() * sizeof(value_type), cudaMemcpyHostToDevice), "cudaMemcpy");
#else
		super::operator=(hv);
#endif
	}

#if defined(__CUDACC__)
    #define DATA_POINTER() thrust::raw_pointer_cast(&super::front())
#elif defined(__OPENCL__)
    #define DATA_POINTER() (pointer)(&super::front()).get_buffer().get()
#else
    #define DATA_POINTER() &super::front()
#endif
	pointer data_pointer(){ return DATA_POINTER(); }
	const_pointer data_pointer() const { return DATA_POINTER(); }
#undef DATA_POINTER
    vector_proxy_type get_proxy(){ return *this; }
    vector_proxy_type get_proxy() const { return *this; }
};

_KIAM_MATH_END
