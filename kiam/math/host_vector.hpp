#pragma once

#include "math_vector.hpp"

#ifdef __CUDACC__
#include <thrust/host_vector.h>
#endif

_KIAM_MATH_BEGIN

template<typename T>
using host_vector_base =
#ifdef __CUDACC__
    thrust::host_vector<T>;
#else
    std::vector<T>;
#endif

template<typename T>
struct host_vector : host_vector_base<T>
{
    typedef T value_type;
    typedef host_vector type;
    typedef host_vector_base<value_type> super;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef vector_proxy<value_type> proxy_type;

    host_vector(){}
    host_vector(size_t size) : super(size){}
    host_vector(size_t size, value_type const& initValue) : super(size, initValue){}
    host_vector(std::initializer_list<T> il) : super(il){}

    explicit host_vector(math_vector<value_type> const& other)
#ifndef __CUDACC__
        : super(other)
#endif
    {
#ifdef __CUDACC__
        if(other.size() > 0){
            super::resize(other.size());
            CUDA_THROW_ON_ERROR(cudaMemcpy(data_pointer(), other.data_pointer(), super::size() * sizeof(value_type), cudaMemcpyDeviceToHost), "cudaMemcpy");
        }
#endif
    }

    host_vector(const host_vector&) = delete;
    void operator=(const host_vector& other) = delete;
    host_vector(host_vector&& other) : super(std::forward<super>(other)){}

    proxy_type get_vector_proxy() const {
        return proxy_type(super::size(), data_pointer());
    }

    host_vector& operator=(math_vector<value_type> const& other){
#ifdef __CUDACC__
        CudaSynchronize sync;
        CUDA_THROW_ON_ERROR(cudaMemcpy(data_pointer(), other.data_pointer(), super::size() * sizeof(value_type), cudaMemcpyDeviceToHost), "cudaMemcpy");
#else
        super::operator=(other);
#endif
        return *this;
    }

    pointer data_pointer(){
#ifdef __CUDACC__
        return thrust::raw_pointer_cast(&super::front());
#else
        return &super::front();
#endif
    }

    const_pointer data_pointer() const {
#ifdef __CUDACC__
        return thrust::raw_pointer_cast(&super::front());
#else
        return &super::front();
#endif
    }
};

_KIAM_MATH_END
