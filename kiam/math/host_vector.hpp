#pragma once

#include "math_vector.hpp"

_KIAM_MATH_BEGIN

template<typename T>
struct host_vector : public std::vector<T>
{
    typedef T value_type;
    typedef host_vector type;
    typedef std::vector<value_type> super;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef vector_proxy<value_type> proxy_type;

    host_vector(){}
    host_vector(size_t size) : super(size){}
    host_vector(size_t size, const value_type &initValue) : super(size, initValue){}
#ifndef DONT_USE_CXX_11
    host_vector(const host_vector&) = delete;
    void operator=(const host_vector &other) = delete;
    host_vector(host_vector &&other) : super(std::forward<super>(other)){}
#endif
    explicit host_vector(const math_vector<value_type>& other)
#ifdef __CUDACC__
    { operator=(other); }
#else
        : super(other){}
#endif

    proxy_type get_vector_proxy() const {
        return proxy_type(super::size(), data_pointer());
    }

    //void operator=(const host_vector &other){ super::operator=(other); }
    void operator=(const math_vector<value_type> &other)
    {
#ifdef __CUDACC__
        super::resize(other.size());
        CUDA_THROW_ON_ERROR(cudaMemcpy(data_pointer(), other.data_pointer(), super::size() * sizeof(value_type), cudaMemcpyDeviceToHost), "cudaMemcpy");
#else
        super::operator=(other);
#endif
    }

    pointer data_pointer(){
        return &super::front();
    }

    const_pointer data_pointer() const {
        return &super::front();
    }
};

_KIAM_MATH_END
