#pragma once

#include "math_def.h"
#include "kiam_math_alg.h"

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

#ifdef CALC_TOTAL_ALLOC_SIZE
extern size_t total_alloc_size;
#define ADD_TOTAL_ALLOC_SIZE(size) add_total_alloc_size(size)
#else
#define ADD_TOTAL_ALLOC_SIZE(size)
#endif

template<typename T>
struct math_vector : public MATH_VECTOR_BASE_CLASS<T>
{
    typedef math_vector type;
    typedef T value_type;
    typedef MATH_VECTOR_BASE_CLASS<value_type> super;
    typedef value_type *pointer;
    typedef const value_type *const_pointer;
    typedef vector_proxy<value_type> vector_proxy_type;

#ifdef __OPENCL__
    math_vector(size_t size = 0, value_type const& init = value_type(), boost::compute::command_queue &queue = boost::compute::system::default_queue()) : super(size, init, queue){
#else
    math_vector(size_t size = 0, value_type const& init = value_type()) : super(size, init){
#endif
        ADD_TOTAL_ALLOC_SIZE(size);
    }

    math_vector(std::initializer_list<T> il) : super(il){}

    void resize(size_t size)
    {
        size_t const old_size = super::size();
        super::resize(size);
        if(old_size == 0)
            ADD_TOTAL_ALLOC_SIZE(size);
    }

    void resize(size_t size, value_type const& init)
    {
        size_t const old_size = super::size();
        super::resize(size, init);
        if(old_size == 0)
            ADD_TOTAL_ALLOC_SIZE(size);
    }

#ifndef DONT_USE_CXX_11
    explicit math_vector(math_vector const& other) : super(other){}
    explicit math_vector(math_vector &&other) : super(std::forward<super>(other)){}
#endif
    explicit math_vector(const host_vector<value_type>& hv) : super(hv){}
//#ifdef __CUDACC__
//  { operator=(hv); }
//#else
//      : super(hv){}
//#endif

    math_vector& operator=(math_vector const& other)
    {
        if(super::size() == 0)
            ADD_TOTAL_ALLOC_SIZE(other.size());
#ifdef __CUDACC__
        super::resize(other.size());
        CUDA_THROW_ON_ERROR(cudaMemcpy(data_pointer(), other.data_pointer(), super::size() * sizeof(value_type), cudaMemcpyDeviceToDevice), "cudaMemcpy");
#else
        super::operator=(other);
#endif
        return *this;
    }
    math_vector& operator=(math_vector &&other){ super::operator=(std::forward<super>(other)); return *this; }

    math_vector& operator=(const host_vector<value_type> &hv)
    {
        if(super::size() == 0)
            ADD_TOTAL_ALLOC_SIZE(hv.size());
#ifdef __CUDACC__
        super::resize(hv.size());
        CUDA_THROW_ON_ERROR(cudaMemcpy(data_pointer(), hv.data_pointer(), super::size() * sizeof(value_type), cudaMemcpyHostToDevice), "cudaMemcpy");
#else
        super::operator=(hv);
#endif
        return *this;
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
    vector_proxy_type get_vector_proxy(){ return *this; }
    vector_proxy_type get_vector_proxy() const { return *this; }

#ifdef CALC_TOTAL_ALLOC_SIZE
    static void add_total_alloc_size(size_t size){
        if(size > 0)
            total_alloc_size += size * sizeof(T);
    }
#endif
};

_KIAM_MATH_END

#include "vector_proxy.hpp"
