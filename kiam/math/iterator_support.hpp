#pragma once

#ifdef __CUDACC__
#include <thrust/memory.h>
#endif

#include "math_def.h"

_KIAM_MATH_BEGIN

template<typename IT>
typename std::iterator_traits<IT>::value_type *get_iterator_data_pointer(IT it){
#ifdef __CUDACC__
	return thrust::raw_pointer_cast(&(*it));
#else
	return &(*it);
#endif
}

template<typename IT>
int get_iterator_stride(const IT &it){
	return 1;
}

template<typename IT> struct stride_iterator;

template<typename IT>
int get_iterator_stride(const stride_iterator<IT> &it){
	return it.stride() * get_iterator_stride(it.inner_iterator());
}

template<typename T> struct proxy_stride_iterator;

template<typename T>
T *get_iterator_data_pointer(proxy_stride_iterator<T> &it){
	return it.data();
}

template<typename T>
const T *get_iterator_data_pointer(const proxy_stride_iterator<T> &it){
	return it.data();
}

template<typename T>
int get_iterator_stride(const proxy_stride_iterator<T> &it){
	return it.stride();
}

_KIAM_MATH_END
