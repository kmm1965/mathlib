#pragma once

#ifdef __CUDACC__
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#endif

#include "math_def.h"

_KIAM_MATH_BEGIN

template<typename T>
struct pinned_host_vector :
#ifdef __CUDACC__
	public thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T> >
#else
	public host_vector<T>
#endif
{
	typedef T value_type;
	typedef pinned_host_vector type;
#ifdef __CUDACC__
	typedef thrust::host_vector<value_type, thrust::system::cuda::experimental::pinned_allocator<value_type> > super;
	typedef value_type *pointer;
	typedef const value_type *const_pointer;
#else
	typedef host_vector<value_type> super;
#endif

	pinned_host_vector(){}
	pinned_host_vector(size_t size) : super(size){}
	pinned_host_vector(size_t size, const value_type &initValue) : super(size, initValue){}
	pinned_host_vector(const pinned_host_vector &other) : super(other){}
	pinned_host_vector(const math_vector<value_type> &other) : super(other){}

	void operator=(const pinned_host_vector &other){ super::operator=(other); }
	void operator=(const math_vector<value_type> &other){ super::operator=(other); }

#ifdef __CUDACC__
	pointer data_pointer(){
		return thrust::raw_pointer_cast(&super::front());
	}

	const_pointer data_pointer() const {
		return thrust::raw_pointer_cast(&super::front());
	}
#endif
};

_KIAM_MATH_END
