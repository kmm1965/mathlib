#pragma once

#include "function_support_fwd.hpp"
#include "iterator_support.hpp"

#include <thrust/pair.h>

#include "math_vector.hpp"
#include "kiam_math_alg.h" // CHECK_CUDA_ERROR

_KIAM_MATH_BEGIN

template<typename T1, typename T2>
#ifdef DONT_USE_CXX_11
struct math_pair : thrust::pair<T1, T2>{};
#else
using math_pair = thrust::pair<T1, T2>;
#endif

template<typename T>
__device__ __host__
void math_swap(T &x, T &y)
{
    T t = x;
    x = y;
    y = t;
}

template<class _FwdIt, class _Ty>
__device__ __host__
void math_fill(_FwdIt _First, _FwdIt _Last, const _Ty& _Val)
{
    for (; _First != _Last; ++_First)
        *_First = _Val;
}

template<class _FwdIt, class _Ty>
__device__ __host__
void math_fill_n(_FwdIt _First, size_t n, const _Ty& _Val){
    math_fill(_First, _First + n, _Val);
}

template<class _InIt, class _OutIt>
__device__ __host__
_OutIt math_copy(_InIt _First, _InIt _Last, _OutIt _Dest)
{
    for (; _First != _Last; ++_Dest, ++_First)
        *_Dest = *_First;
    return _Dest;
}

template<class _InIt, class _OutIt>
__device__ __host__
_OutIt math_copy_n(_InIt _First, size_t n, _OutIt _Dest){
    return math_copy(_First, _First + n, _Dest);
}

template<class _InIt, class _Ty, class _Fn2>
__device__ __host__
_Ty math_accumulate(_InIt _First, _InIt _Last, _Ty _Val, _Fn2 _Func)
{
    for (; _First != _Last; ++_First)
        _Val = _Func(_Val, *_First);
    return _Val;
}

template<class _InIt, class _Ty, class _Fn2>
__device__ __host__
_Ty math_accumulate_n(_InIt _First, size_t n, _Ty _Val, _Fn2 _Func){
    return math_accumulate(_First, _First + n, _Val, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
__device__ __host__
_OutIt math_transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func)
{
    for (; _First != _Last; ++_First, ++_Dest)
        *_Dest = _Func(*_First);
    return _Dest;
}

template<class _InIt, class _OutIt, class _Fn1>
__device__ __host__
_OutIt math_transform_n(_InIt _First, size_t n, _OutIt _Dest, _Fn1 _Func){
    return math_transform(_First, _First + n, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
__device__ __host__
_OutIt math_transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func)
{
    for (; _First1 != _Last1; ++_First1, ++_First2, ++_Dest)
        *_Dest = _Func(*_First1, *_First2);
    return (_Dest);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
__device__ __host__
_OutIt math_transform_n(_InIt1 _First1, size_t n, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func){
    return math_transform(_First1, _First1 + n, _First2, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
__device__ __host__
_Ty math_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2)
{
    for (; _First1 != _Last1; ++_First1, ++_First2)
        _Val = _Func1(_Val, _Func2(*_First1, *_First2));
    return (_Val);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
__device__ __host__
_Ty math_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2){
    return math_inner_product(_First1, _First1 + n, _First2, _Val, _Func1, _Func2);
}

template<class _InIt1, class _InIt2, class _Ty>
__device__ __host__
_Ty math_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val){
    return math_inner_product(_First1, _Last1, _First2, _Val, thrust::plus<_Ty>(), thrust::multiplies<_Ty>());
}

template<class _InIt1, class _InIt2, class _Ty>
__device__ __host__
_Ty math_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val){
    return math_inner_product(_First1, _First1 + n, _First2, _Val);
}

template<typename T, typename V, class BO>
__global__
void reduce_g1(const T *data, size_t n, V r, size_t stride, BO bin_op, V *result)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t j = i, threads = gridDim.x * blockDim.x; j < n; j += threads)
        r = bin_op(r, data[j * stride]);
    result[i] = r;
}

template<typename V, class BO>
__global__
void reduce_g_final(size_t threads, size_t size, BO bin_op, V *result)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i + threads < size)
        result[i] = bin_op(result[i], result[i + threads]);
}

template<typename T, typename V, class BO1, class BO2>
V reduce_n(const T *data, size_t n, int stride, V init, BO1 bin_op1, BO2 bin_op2, math_vector<V> &work)
{
    if(n == 0)
        return init;
    int deviceNum;
    cudaError_t error_id;
    if((error_id = cudaGetDevice(&deviceNum)) != cudaSuccess)
        throw thrust::system_error(error_id, thrust::cuda_category());
    cudaDeviceProp deviceProp;
    if((error_id = cudaGetDeviceProperties(&deviceProp, deviceNum)) != cudaSuccess)
        throw thrust::system_error(error_id, thrust::cuda_category());
    size_t dimGrid = deviceProp.multiProcessorCount;
    size_t dimBlock = deviceProp.maxThreadsPerBlock;
    size_t size = dimGrid * dimBlock;
    if(size > n){
        if((dimGrid = n / dimBlock) == 0){
            dimGrid = 1;
            size = dimBlock = n;
        } else size = dimGrid * dimBlock;
    }
    if (work.size() < size)
        work.resize(size);
    V *work_data = work.data_pointer();
    reduce_g1<<<dimGrid, dimBlock>>>(data, n, init, stride, bin_op1, work_data);
    CHECK_CUDA_ERROR("reduce_g1");
    while(size > 1){
        size_t new_size = (size + 1) / 2;
        if((dimGrid = (new_size + dimBlock - 1) / dimBlock) == 1)
            dimBlock = new_size;
        reduce_g_final<<<dimGrid, dimBlock>>>(new_size, size, bin_op2, work_data);
        CHECK_CUDA_ERROR("reduce_g_final");
        size = new_size;
    }
    return work.front();
}

template<typename T, typename V, class BO1, class BO2>
V reduce_n(const T *data, size_t n, int stride, V init, BO1 bin_op1, BO2 bin_op2)
{
    math_vector<V> work;
    return reduce_n(data, n, stride, init, bin_op1, bin_op2, work);
}

template<typename IT, typename V, class BO1, class BO2>
V reduce_iterator(IT it, size_t n, V init, BO1 bin_op1, BO2 bin_op2){
    return reduce_n(get_iterator_data_pointer(it), n, get_iterator_stride(it), init, bin_op1, bin_op2);
}

template<typename IT, typename V, class BO1, class BO2>
V reduce_iterator(IT it, size_t n, V init, BO1 bin_op1, BO2 bin_op2, math_vector<typename std::iterator_traits<IT>::value_type> &work){
    return reduce_n(get_iterator_data_pointer(it), n, get_iterator_stride(it), init, bin_op1, bin_op2, work);
}

template<typename T, typename V, class BO1, class BO2>
V reduce_vector(const math_vector<T> &v, V init, BO1 bin_op1, BO2 bin_op2){
    return reduce_n(v.data_pointer(), v.size(), 1, init, bin_op1, bin_op2);
}

template<typename T, typename V, class BO1, class BO2>
V reduce_vector(const math_vector<T> &v, V init, BO1 bin_op1, BO2 bin_op2, math_vector<T> &work){
    return reduce_n(v.data_pointer(), v.size(), 1, init, bin_op1, bin_op2, work);
}

struct math_string
{
    __DEVICE __HOST
    math_string(const char *s)
    {
        unsigned i;
        for (i = 0; i < sizeof data - 1 && s[i]; ++i)
            data[i] = s[i];
        data[i] = 0;
    }

    __DEVICE __HOST
    unsigned length() const
    {
        unsigned i = 0;
        for (i = 0; data[i]; ++i);
        return i;
    }

    __DEVICE __HOST
    bool operator==(const char *s) const
    {
        unsigned i = 0;
        for (i = 0; data[i]; ++i)
            if (data[i] != s[i])
                return false;
        return s[i] == 0;
    }

    __DEVICE __HOST
    bool operator!=(const char *s) const {
        return !operator==(s);
    }

private:
    char data[10]; // short strings only
};

_KIAM_MATH_END
