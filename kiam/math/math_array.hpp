#pragma once

#include "math_def.h"

#ifndef __CUDACC__
#include <array>
#endif  // __CUDACC__

_KIAM_MATH_BEGIN

#ifdef __CUDACC__

template<typename T, size_t N>
struct math_array
{
    typedef T value_type;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef T *pointer;
    typedef const T *const_pointer;
    typedef pointer iterator;
    typedef const_pointer const_iterator;

    __device__ __host__
    size_t size() const {
        return N;
    }

    __device__ __host__
    pointer data(){
        return m_data;
    }

    __device__ __host__
    const_pointer data() const {
        return m_data;
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

    value_type m_data[N];
};

#else   // __CUDACC__

template<typename T, size_t N>
using math_array = std::array<T, N>;

#endif  // __CUDACC__

_KIAM_MATH_END
