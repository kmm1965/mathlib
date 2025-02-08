#pragma once

#include "math_mpl.hpp"     // math_less
#include "kiam_math_alg.h"  // CHECK_CUDA_ERROR

#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>

#if THRUST_VERSION < 200000
#define THRUST_DEVICE
#else
#define THRUST_DEVICE thrust::device,
#endif

_KIAM_MATH_BEGIN

template<typename T1, typename T2>
#ifdef DONT_USE_CXX_11
struct math_pair : thrust::pair<T1, T2>{};
#else
using math_pair = thrust::pair<T1, T2>;
#endif

template<class Arg, class Result>
struct math_unary_function
{
    typedef Arg argument_type;
    typedef Result result_type;
};

template<class Arg1, class Arg2, class Result>
struct math_binary_function
{
    typedef Arg1 first_argument_type;
    typedef Arg2 second_argument_type;
    typedef Result result_type;
};

template<typename T>
__device__ __host__
void math_swap(T &x, T &y)
{
    T t = x;
    x = y;
    y = t;
}

template<typename T>
__device__ __host__
T math_abs(T const& x) noexcept {
    return x >= 0 ? x : -x;
}

template<class _FwdIt, class _Ty>
__device__ __host__
void math_fill(_FwdIt _First, _FwdIt _Last, _Ty const& _Val)
{
    for(; _First != _Last; ++_First)
        *_First = _Val;
}
template<class _FwdIt, class _Ty>
__device__ __host__
void math_fill_n(_FwdIt _First, size_t n, _Ty const& _Val){
    math_fill(_First, _First + n, _Val);
}

template<class _FwdIt, class _Ty>
void device_fill(_FwdIt _First, _FwdIt _Last, _Ty const& _Val){
    CudaSynchronize sync;
    thrust::fill(THRUST_DEVICE _First, _Last, _Val);
}
template<class _FwdIt, class _Ty>
void device_fill_n(_FwdIt _First, size_t n, _Ty const& _Val){
    CudaSynchronize sync;
    thrust::fill_n(THRUST_DEVICE _First, n, _Val);
}

template<class _InIt, class _OutIt>
__device__ __host__
_OutIt math_copy(_InIt _First, _InIt _Last, _OutIt _Dest)
{
    for(; _First != _Last; ++_Dest, ++_First)
        *_Dest = *_First;
    return _Dest;
}
template<class _InIt, class _OutIt>
__device__ __host__
_OutIt math_copy_n(_InIt _First, size_t n, _OutIt _Dest){
    return math_copy(_First, _First + n, _Dest);
}

template<class _InIt, class _OutIt>
_OutIt device_copy_(_InIt _First, _InIt _Last, _OutIt _Dest){
    CudaSynchronize sync;
    return thrust::copy(THRUST_DEVICE _First, _Last, _Dest);
}

template<class _InIt, class _OutIt>
_OutIt device_copy_n_(_InIt _First, size_t n, _OutIt _Dest){
    CudaSynchronize sync;
    return thrust::copy_n(THRUST_DEVICE _First, n, _Dest);
}

template<typename T>
T* device_copy_n(const T *_First, size_t n, T *_Dest){
    CUDA_THROW_ON_ERROR(cudaMemcpy(_Dest, _First, n * sizeof(T), cudaMemcpyDeviceToDevice), "cudaMemcpy");
    return _Dest + n;
}

template<typename T>
T* host_to_device_copy_n(const T *_First, size_t n, T *_Dest){
    //return thrust::raw_pointer_cast(thrust::copy_n(_First, n, thrust::device_ptr<T>_Dest));
    CUDA_THROW_ON_ERROR(cudaMemcpy(_Dest, _First, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy");
    return _Dest + n;
}

template<typename T>
T* device_to_host_copy_n(const T* _First, size_t n, T* _Dest){
    //return thrust::copy_n(thrust::device_ptr<T>((T*)_First), n, _Dest);
    CUDA_THROW_ON_ERROR(cudaMemcpy(_Dest, _First, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy");
    return _Dest + n;
}

template<class _InIt, class _Ty>
__device__ __host__
_Ty math_reduce(_InIt _First, _InIt _Last, _Ty _Val)
{
    for(_InIt _First_ = _First; _First_ != _Last; ++_First_)
        _Val += *_First_;
    return _Val;
}

template<class _InIt, class _Ty>
__device__ __host__
_Ty math_reduce_n(_InIt _First, size_t n, _Ty _Val){
    return math_reduce(_First, _First + n, _Val);
}

template<class _InIt, class _Ty, class _BinOp>
__device__ __host__
_Ty math_reduce(_InIt _First, _InIt _Last, _Ty _Val, _BinOp _Reduce_op)
{
    for(_InIt _First_ = _First; _First_ != _Last; ++_First_)
        _Val = _Reduce_op(_Val, *_First_);
    return _Val;
}

template<class _InIt, class _Ty, class _BinOp>
__device__ __host__
_Ty math_reduce_n(_InIt _First, size_t n, _Ty _Val, _BinOp _Reduce_op){
    return math_reduce(_First, _First + n, _Val, _Reduce_op);
}

template<class _InIt, class _Ty>
_Ty device_reduce(_InIt _First, _InIt _Last, _Ty _Val){
    CudaSynchronize sync;
    return thrust::reduce(THRUST_DEVICE _First, _Last, _Val);
}

template<class _InIt, class _Ty>
_Ty device_reduce_n(_InIt _First, size_t n, _Ty _Val){
    return device_reduce(_First, _First + n, _Val);
}

template<class _InIt, class _Ty, class _BinOp>
_Ty device_reduce(_InIt _First, _InIt _Last, _Ty _Val, _BinOp _Reduce_op){
    CudaSynchronize sync;
    return thrust::reduce(THRUST_DEVICE _First, _Last, _Val, _Reduce_op);
}

template<class _InIt, class _Ty, class _BinOp>
_Ty device_reduce_n(_InIt _First, size_t n, _Ty _Val, _BinOp _Reduce_op){
    return device_reduce(_First, _First + n, _Val, _Reduce_op);
}

template<class _InIt, class _OutIt, class _Fn1>
__device__ __host__
_OutIt math_transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func)
{
    for(; _First != _Last; ++_First, ++_Dest)
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
    for(; _First1 != _Last1; ++_First1, ++_First2, ++_Dest)
        *_Dest = _Func(*_First1, *_First2);
    return _Dest;
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
__device__ __host__
_OutIt math_transform_n(_InIt1 _First1, size_t n, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func){
    return math_transform(_First1, _First1 + n, _First2, _Dest, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt device_transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func){
    CudaSynchronize sync;
    return thrust::transform(THRUST_DEVICE _First, _Last, _Dest, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt device_transform_n(_InIt _First, size_t n, _OutIt _Dest, _Fn1 _Func){
    return device_transform(_First, _First + n, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt device_transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func){
    CudaSynchronize sync;
    return thrust::transform(THRUST_DEVICE _First1, _Last1, _First2, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt device_transform_n(_InIt1 _First1, size_t n, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func){
    return device_transform(_First1, _First1 + n, _First2, _Dest, _Func);
}

template<class _FwdIt, class _Ty, class _BinOp, class _UnaryOp>
__device__ __host__
_Ty math_transform_reduce(_FwdIt _First, _FwdIt _Last, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op)
{
    for(; _First != _Last; ++_First)
        _Val= _Reduce_op(_Val, _Transform_op(*_First));
    return _Val;
}

template<class _FwdIt, class _Ty, class _BinOp, class _UnaryOp>
__device__ __host__
_Ty math_transform_reduce_n(_FwdIt _First, size_t n, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op){
    return math_transform_reduce(_First, _First + n, _Val, _Reduce_op, _Transform_op);
}

template<class _FwdIt1, class _FwdIt2, class _Ty, class _ReduceBinOp, class _TransformBinOp>
__device__ __host__
_Ty math_transform_reduce(_FwdIt1 _First1, _FwdIt1 _Last1, _FwdIt2 _First2, _Ty _Val, _ReduceBinOp _Reduce_op, _TransformBinOp _Transform_op)
{
    for(; _First1 != _Last1; ++_First1, ++_First2)
        _Val = _Reduce_op(_Val, _Transform_op(*_First1, *_First2));
    return _Val;
}

template<class _FwdIt1, class _FwdIt2, class _Ty, class _ReduceBinOp, class _TransformBinOp>
__device__ __host__
_Ty math_transform_reduce_n(_FwdIt1 _First1, size_t n, _FwdIt2 _First2, _Ty _Val, _ReduceBinOp _Reduce_op, _TransformBinOp _Transform_op){
    return math_transform_reduce(_First1, _First1 + n, _First2, _Val, _Reduce_op, _Transform_op);
}

template<class _FwdIt, class _Ty, class _BinOp, class _UnaryOp>
_Ty device_transform_reduce(_FwdIt _First, _FwdIt _Last, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op){
    CudaSynchronize sync;
    return thrust::transform_reduce(THRUST_DEVICE _First, _Last, _Transform_op, _Val, _Reduce_op);
}

template<class _FwdIt, class _Ty, class _BinOp, class _UnaryOp>
_Ty device_transform_reduce_n(_FwdIt _First, size_t n, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op){
    return device_transform_reduce(_First, _First + n, _Val, _Reduce_op, _Transform_op);
}

template<class _FwdIt1, class _FwdIt2, class _Ty, class _BinOp, class _UnaryOp>
_Ty device_transform_reduce(_FwdIt1 _First1, _FwdIt1 _Last1, _FwdIt2 _First2, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op){
    CudaSynchronize sync;
    return thrust::transform_reduce(THRUST_DEVICE _First1, _Last1, _First2, _Transform_op, _Val, _Reduce_op);
}

template<class _FwdIt1, class _FwdIt2, class _Ty, class _BinOp, class _UnaryOp>
_Ty device_transform_reduce_n(_FwdIt1 _First1, size_t n, _FwdIt2 _First2, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op){
    return device_transform_reduce(_First1, _First1 + n, _First2, _Val, _Reduce_op, _Transform_op);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
__device__ __host__
_Ty math_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2)
{
    for(; _First1 != _Last1; ++_First1, ++_First2)
        _Val = _Func1(_Val, _Func2(*_First1, *_First2));
    return _Val;
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

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
_Ty device_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2){
    CudaSynchronize sync;
    return thrust::inner_product(THRUST_DEVICE _First1, _Last1, _First2, _Val, _Func1, _Func2);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
_Ty device_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2){
    return device_inner_product(_First1, _First1 + n, _First2, _Val, _Func1, _Func2);
}

template<class _InIt1, class _InIt2, class _Ty>
_Ty device_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val){
    CudaSynchronize sync;
    return thrust::inner_product(THRUST_DEVICE _First1, _Last1, _First2, _Val);
}

template<class _InIt1, class _InIt2, class _Ty>
_Ty device_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val){
    return device_inner_product(_First1, _First1 + n, _First2, _Val);
}

template<class _InIt, class _Fn1>
__device__ __host__
void math_for_each(_InIt _First, _InIt _Last, _Fn1 _Func){
    for(; _First != _Last; ++_First)
        _Func(*_First);
}

template<class _InIt, class _Fn1>
__device__ __host__
void math_for_each_n(_InIt _First, size_t n, _Fn1 _Func){
    math_for_each(_First, _First + n, _Func);
}

template<class _InIt, class _Fn1>
void device_for_each(_InIt _First, _InIt _Last, _Fn1 _Func){
    CudaSynchronize sync;
    thrust::for_each(THRUST_DEVICE _First, _Last, _Func);
}

template<class _InIt, class _Fn1>
void device_for_each_n(_InIt _First, size_t n, _Fn1 _Func){
    CudaSynchronize sync;
    thrust::for_each_n(THRUST_DEVICE _First, n, _Func);
}

template<class _FwdIt>
__device__ __host__ _FwdIt math_max_element(_FwdIt _First, _FwdIt _Last)
{
    _FwdIt _Found = _First;
    if(_First != _Last){
        while(++_First != _Last){
            if(*_Found < *_First)
                _Found = _First;
        }
    }
    return _Found;
}

template<class _FwdIt>
__device__ __host__ _FwdIt math_max_element_n(_FwdIt _First, size_t n){
    return math_max_element(_First, _First + n);
}

template<class _FwdIt, class _Compare>
__device__ __host__ _FwdIt math_max_element(_FwdIt _First, _FwdIt _Last, _Compare _Comp)
{
    _FwdIt _Found = _First;
    if(_First != _Last){
        while(++_First != _Last){
            if(_Comp(*_Found, *_First))
                _Found = _First;
        }
    }
    return _Found;
}

template<class _FwdIt, class _Compare>
__device__ __host__ _FwdIt math_max_element_n(_FwdIt _First, size_t n, _Compare _Comp){
    return math_max_element(_First, _First + n, _Comp);
}

template<class _FwdIt>
_FwdIt device_max_element(_FwdIt _First, _FwdIt _Last){
    return thrust::max_element(THRUST_DEVICE _First, _Last);
}

template<class _FwdIt>
_FwdIt device_max_element_n(_FwdIt _First, size_t n){
    return device_max_element(_First, _First + n);
}

template<class _FwdIt, class _Compare>
_FwdIt device_max_element(_FwdIt _First, _FwdIt _Last, _Compare _Comp){
    return thrust::max_element(THRUST_DEVICE _First, _Last, _Comp);
}

template<class _FwdIt, class _Compare>
_FwdIt device_max_element_n(_FwdIt _First, size_t n, _Compare _Comp){
    return device_max_element(_First, _First + n, _Comp);
}

template<class _FwdIt>
__device__ __host__ _FwdIt math_min_element(_FwdIt _First, _FwdIt _Last)
{
    _FwdIt _Found = _First;
    if(_First != _Last){
        while(++_First != _Last){
            if(*_First < *_Found)
                _Found = _First;
        }
    }
    return _Found;
}

template<class _FwdIt>
__device__ __host__ _FwdIt math_min_element_n(_FwdIt _First, size_t n){
    return math_min_element(_First, _First + n);
}

template<class _FwdIt, class _Compare>
__device__ __host__ _FwdIt math_min_element(_FwdIt _First, _FwdIt _Last, _Compare _Comp)
{
    _FwdIt _Found = _First;
    if(_First != _Last){
        while(++_First != _Last){
            if(_Comp(*_First, *_Found))
                _Found = _First;
        }
    }
    return _Found;
}

template<class _FwdIt, class _Compare>
__device__ __host__ _FwdIt math_min_element_n(_FwdIt _First, size_t n, _Compare _Comp){
    return math_min_element(_First, _First + n, _Comp);
}

template<class _FwdIt>
_FwdIt device_min_element(_FwdIt _First, _FwdIt _Last){
    return thrust::min_element(THRUST_DEVICE _First, _Last);
}

template<class _FwdIt>
_FwdIt device_min_element_n(_FwdIt _First, size_t n){
    return device_min_element(_First, _First + n);
}

template<class _FwdIt, class _Compare>
_FwdIt device_min_element(_FwdIt _First, _FwdIt _Last, _Compare _Comp){
    return thrust::min_element(THRUST_DEVICE _First, _Last, _Comp);
}

template<class _FwdIt, class _Compare>
_FwdIt device_min_element_n(_FwdIt _First, size_t n, _Compare _Comp){
    return device_min_element(_First, _First + n, _Comp);
}

template<class _InIt, class _Ty>
__device__ __host__ _InIt math_find(_InIt _First, const _InIt _Last, const _Ty& _Val)
{
    for(; _First != _Last && !(*_First == _Val); ++_First);
    return _First;
}

template<class _InIt, class _Ty>
__device__ __host__ _InIt math_find_n(_InIt _First, size_t n, const _Ty& _Val){
    return math_find(_First, _First + n, _Val);
}
#if 0
template<class _InIt, class _Ty>
_InIt device_find(_InIt _First, const _InIt _Last, const _Ty& _Val){
    CudaSynchronize sync;
    return thrust::find(THRUST_DEVICE _First, _Last, _Val);
}
template<class _InIt, class _Ty>
_InIt device_find_n(_InIt _First, size_t n, const _Ty& _Val){
    return device_find(_First, _First + n, _Val);
}
#endif // 0

template<class _InIt, class _Pr>
__device__ __host__ _InIt math_find_if(_InIt _First, const _InIt _Last, _Pr _Pred)
{
    for(; _First != _Last && !_Pred(*_First); ++_First);
    return _First;
}
template<class _InIt, class _Pr>
__device__ __host__ _InIt math_find_if_n(_InIt _First, size_t n, _Pr _Pred){
    return math_find_if(_First, _First + n, _Pred);
}

#if 0
template<class _InIt, class _Pr>
_InIt device_find_if(_InIt _First, const _InIt _Last, _Pr _Pred){
    CudaSynchronize sync;
    return thrust::find_if(THRUST_DEVICE _First, _Last, _Pred);
}
template<class _InIt, class _Pr>
_InIt device_find_if_n(_InIt _First, size_t n, _Pr _Pred){
    return device_find_if(_First, _First + n, _Pred);
}
#endif // 0

template <class _FwdIt, class _Ty, class _Pr>
__device__ __host__ _FwdIt math_lower_bound(_FwdIt _First, const _FwdIt _Last, const _Ty& _Val, _Pr _Pred)
{
    auto _Count = _Last - _First;

    while(0 < _Count){ // divide and conquer, find half that contains answer
        const decltype(_Count) _Count2 = _Count / 2;
        const auto _Mid = _First + _Count2;
        if(_Pred(*_Mid, _Val)){ // try top half
            _First = _Mid + 1;
            _Count -= _Count2 + 1;
        } else _Count = _Count2;
    }
    return _First;
}

template <class _FwdIt, class _Ty, class _Pr>
__device__ __host__ _FwdIt math_lower_bound_n(_FwdIt _First, size_t n, const _Ty& _Val, _Pr _Pred){
    return math_lower_bound(_First, _First + n, _Val, _Pred);
}

template <class _FwdIt, class _Ty>
__device__ __host__ _FwdIt math_lower_bound(_FwdIt _First, _FwdIt _Last, const _Ty& _Val){
    return math_lower_bound(_First, _Last, _Val, math_less<_Ty>());
}

template <class _FwdIt, class _Ty>
__device__ __host__ _FwdIt math_lower_bound_n(_FwdIt _First, size_t n, const _Ty& _Val){
    return math_lower_bound(_First, _First + n, _Val);
}

template <class _FwdIt, class _Ty, class _Pr>
__device__ __host__ _FwdIt math_upper_bound(_FwdIt _First, const _FwdIt _Last, const _Ty& _Val, _Pr _Pred)
{
    auto _Count = _Last - _First;

    while(0 < _Count){ // divide and conquer, find half that contains answer
        const decltype(_Count) _Count2 = _Count / 2;
        const auto _Mid = _First + _Count2;
        if(_Pred(_Val, *_Mid)){
            _Count = _Count2;
        } else { // try top half
            _First = _Mid + 1;
            _Count -= _Count2 + 1;
        }
    }
    return _First;
}

template <class _FwdIt, class _Ty, class _Pr>
__device__ __host__ _FwdIt math_upper_bound_n(_FwdIt _First, size_t n, const _Ty& _Val, _Pr _Pred){
    return math_upper_bound(_First, _First + n, _Val, _Pred);
}

template <class _FwdIt, class _Ty>
__device__ __host__ _FwdIt math_upper_bound(_FwdIt _First, _FwdIt _Last, const _Ty& _Val){
    return math_upper_bound(_First, _Last, _Val, math_less<_Ty>());
}

template <class _FwdIt, class _Ty>
__device__ __host__ _FwdIt math_upper_bound_n(_FwdIt _First, size_t n, const _Ty& _Val){
    return math_upper_bound(_First, _First + n, _Val);
}

template <class _FwdIt, class _Fn>
void math_generate(_FwdIt _First, _FwdIt _Last, _Fn _Func){
    for(; _First != _Last; ++_First)
        *_First = _Func();
}

template <class _FwdIt, class _Fn>
void math_generate_n(_FwdIt _First, size_t n, _Fn _Func){
    math_generate(_First, _First + n, _Func);
}

template <class _FwdIt, class _Fn>
void device_generate(_FwdIt _First, _FwdIt _Last, _Fn _Func){
    thrust::generate(THRUST_DEVICE _First, _Last, _Func);
}

template <class _FwdIt, class _Fn>
void device_generate_n(_FwdIt _First, size_t n, _Fn _Func){
    device_generate(_First, _First + n, _Func);
}

template <class _FwdIt>
void math_sequence(_FwdIt _First, _FwdIt _Last){
    size_t i = 0;
    for(; _First != _Last; ++_First, ++i)
        *_First = i;
}

template <class _FwdIt>
void math_sequence_n(_FwdIt _First, size_t n){
    math_sequence(_First, _First + n);
}

template <class _FwdIt>
void device_sequence(_FwdIt _First, _FwdIt _Last){
    thrust::sequence(_First, _Last);
}

template <class _FwdIt>
void device_sequence_n(_FwdIt _First, size_t n){
    device_sequence(_First, _First + n);
}

template<typename F>
__global__ void g_execute_once(F f){
    f();
}

template<typename F>
void execute_once(F f){
    g_execute_once<<<1,1>>>(f);
    CUDA_THROW_ON_ERROR(cudaGetLastError(), "g_execute_once");
    //CHECK_CUDA_ERROR("g_execute_once");
}

template<typename F, typename... Args>
__global__ void g_execute_n(F f, size_t n, Args... args){
    size_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        f(i, args...);
}

template<typename F, typename... Args>
void execute_n(size_t n, F f, Args... args){
    unsigned const
        dimGrid = unsigned((n + BLOCK1_SIZE - 1) / BLOCK1_SIZE),
        dimBlock = unsigned(n <= BLOCK1_SIZE ? n : BLOCK1_SIZE);
    g_execute_n<<<dimGrid, dimBlock>>>(f, n, args...);
    CUDA_THROW_ON_ERROR(cudaGetLastError(), "g_execute_n");
    //CHECK_CUDA_ERROR("g_execute_n");
}

struct math_string
{
    __device__ __host__
    math_string(){
        data[0] = 0;
    }

    __device__ __host__
    math_string(const char* s){
        *this = s;
    }

    __device__ __host__
    math_string(math_string const& other){
        *this = other.c_str();
    }

    math_string(std::string const& s){
        *this = s.c_str();
    }

    __device__ __host__
    math_string& operator=(const char* s)
    {
        unsigned i;
        for(i = 0; i < sizeof data - 1 && s[i]; ++i)
            data[i] = s[i];
        data[i] = 0;
        return *this;
    }

    __device__ __host__
    unsigned length() const
    {
        unsigned i;
        for(i = 0; data[i]; ++i);
        return i;
    }

    __device__ __host__
    unsigned size() const {
        return length();
    }

    __device__ __host__
    bool empty() const {
        return length() == 0;
    }

    __device__ __host__
    const char *c_str() const {
        return data;
    }

    __device__ __host__
    char operator[](size_t idx) const
    {
#ifndef __CUDACC__
        assert(idx < length());
#endif
        return data[idx];
    }

    __device__ __host__
    math_string& operator+=(const char* s)
    {
        unsigned i = length();
        for(unsigned j = 0; i < sizeof data - 1 && s[j]; ++i, ++j)
            data[i] = s[j];
        data[i] = 0;
        return *this;
    }

    __device__ __host__
    math_string& operator+=(math_string const& other){
        return *this += other.data;
    }

    __device__ __host__
    bool operator==(const char *s) const
    {
        unsigned i = 0;
        for(i = 0; data[i]; ++i)
            if (data[i] != s[i])
                return false;
        return s[i] == 0;
    }

    __device__ __host__
    bool operator!=(const char *s) const {
        return !operator==(s);
    }

private:
    char data[100]; // short strings only
};

__device__ __host__
inline math_string operator+(math_string const& str1, math_string const& str2)
{
    math_string result(str1);
    result += str2;
    return result;
}

__device__ __host__
inline math_string operator+(const char* s, math_string const& str)
{
    math_string result(s);
    result += str;
    return result;
}

__device__ __host__
inline math_string operator+(math_string const& str, const char* s)
{
    math_string result(str);
    result += s;
    return result;
}

template<class Incrementable>
using math_counting_iterator = thrust::counting_iterator<Incrementable>;

template<typename... Types>
using math_tuple = thrust::tuple<Types...>;

template<typename... Types>
__host__ __device__
thrust::tuple<Types...> make_math_tuple(Types&&... args){
    return thrust::make_tuple(args...);
}

template<size_t _Index, class _Tuple>
using math_tuple_element = thrust::tuple_element<_Index, _Tuple>;

template<size_t _Index, class _Tuple>
using math_tuple_element_t = typename math_tuple_element<_Index, _Tuple>::type;

template<class _Tuple>
using math_tuple_size = thrust::tuple_size<_Tuple>;

template<size_t _Index, typename... _Types>
__host__ __device__
constexpr typename thrust::tuple_element<_Index, math_tuple<_Types...>>::type& math_get(math_tuple<_Types...> &t){
    return thrust::get<_Index>(t);
}

template<size_t _Index, typename... _Types>
__host__ __device__
constexpr typename thrust::tuple_element<_Index, math_tuple<_Types...>>::type const& math_get(math_tuple<_Types...> const& t){
    return thrust::get<_Index>(t);
}

template<typename T>
thrust::device_ptr<T> make_device_ptr(T* p){
    return thrust::device_ptr<T>(p);
}

template<typename T>
T* make_ptr(thrust::device_ptr<T> &p){
    return thrust::raw_pointer_cast(p);
}

template<typename T>
const T* make_ptr(thrust::device_ptr<T> const& p){
    return thrust::raw_pointer_cast(p);
}

_KIAM_MATH_END
