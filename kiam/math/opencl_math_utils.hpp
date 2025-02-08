#pragma once

#include "math_def.h"

#include <boost/compute/algorithm/iota.hpp>

_KIAM_MATH_BEGIN

template<typename T> inline
void math_swap(T &x, T &y)
{
    T t = x;
    x = y;
    y = t;
}

template<class _T, class _Ty> inline
void math_fill(_T *_First, _T *_Last, const _Ty& _Val)
{
    for (; _First != _Last; ++_First)
        *_First = _Val;
}

template<class _T, class _Ty> inline
void math_fill_n(_T *_First, size_t n, const _Ty& _Val){
    math_fill(_First, _First + n, _Val);
}

template<class _InT, class _OutT> inline
_OutT *math_copy(const _InT *_First, const _InT *_Last, _OutT *_Dest)
{
    for (; _First != _Last; ++_Dest, ++_First)
        *_Dest = *_First;
    return _Dest;
}

template<class _InT, class _OutT> inline
_OutT math_copy_n(const _InT *_First, size_t n, _OutT *_Dest){
    return math_copy(_First, _First + n, _Dest);
}

template<class _InT, class _Ty, class _Fn2> inline
_Ty math_reduce(const _InT *_First, const _InT *_Last, _Ty _Val, _Fn2 _Func)
{
    for (; _First != _Last; ++_First)
        _Val = _Func(_Val, *_First);
    return _Val;
}

template<class _InT, class _Ty, class _Fn2> inline
_Ty math_reduce_n(const _InT *_First, size_t n, _Ty _Val, _Fn2 _Func){
    return math_reduce(_First, _First + n, _Val, _Func);
}

template<class _InT, class _OutT, class _Fn1> inline
_OutT math_transform(const _InT *_First, const _InT *_Last, _OutT *_Dest, _Fn1 _Func)
{
    for (; _First != _Last; ++_First, ++_Dest)
        *_Dest = _Func(*_First);
    return _Dest;
}

template<class _InT, class _OutT, class _Fn1> inline
_OutT math_transform_n(const _InT *_First, size_t n, _OutT *_Dest, _Fn1 _Func){
    return math_transform(_First, _First + n, _Dest, _Func);
}

template<class _InT1, class _InT2, class _OutT, class _Fn2> inline
_OutT math_transform(const _InT1 *_First1, const _InT1 *_Last1, const _InT2 *_First2, _OutT *_Dest, _Fn2 _Func)
{
    for (; _First1 != _Last1; ++_First1, ++_First2, ++_Dest)
        *_Dest = _Func(*_First1, *_First2);
    return (_Dest);
}

template<class _InT1, class _InT2, class _OutT, class _Fn2> inline
_OutT math_transform_n(const _InT1 *_First1, size_t n, const _InT2 *_First2, _OutT *_Dest, _Fn2 _Func){
    return math_transform(_First1, _First1 + n, _First2, _Dest, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt device_transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func){
    OpenCLSynchronize sync;
    return boost::compute::transform(_First, _Last, _Dest, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt device_transform_n(_InIt _First, size_t n, _OutIt _Dest, _Fn1 _Func){
    return device_transform(_First, _First + n, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt device_transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func, boost::compute::command_queue &queue = boost::compute::system::default_queue()){
    OpenCLSynchronize sync;
    return boost::compute::transform(_First1, _Last1, _First2, _Dest, _Func, queue);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt device_transform_n(_InIt1 _First1, size_t n, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func, boost::compute::command_queue& queue = boost::compute::system::default_queue()){
    return device_transform(_First1, _First1 + n, _First2, _Dest, _Func, queue);
}

template<class _InT1, class _InT2, class _Ty, class _Fn21, class _Fn22> inline
_Ty math_inner_product(const _InT1 *_First1, const _InT1 *_Last1, const _InT2 *_First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2)
{
    for (; _First1 != _Last1; ++_First1, ++_First2)
        _Val = _Func1(_Val, _Func2(*_First1, *_First2));
    return (_Val);
}

template<class _InT1, class _InT2, class _Ty, class _Fn21, class _Fn22> inline
_Ty math_inner_product_n(const _InT1 *_First1, size_t n, const _InT2 *_First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2){
    return math_inner_product(_First1, _First1 + n, _First2, _Val, _Func1, _Func2);
}

template<class _InT1, class _InT2, class _Ty> inline
_Ty math_inner_product(const _InT1 *_First1, const _InT1 *_Last1, const _InT2 *_First2, _Ty _Val){
    return math_inner_product(_First1, _Last1, _First2, _Val, boost::compute::plus<_Ty>(), boost::compute::multiplies<_Ty>());
}

template<class _InT1, class _InT2, class _Ty> inline
_Ty math_inner_product_n(const _InT1 *_First1, size_t n, const _InT2 *_First2, _Ty _Val){
    return math_inner_product(_First1, _First1 + n, _First2, _Val);
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
void device_sequence(_FwdIt _First, _FwdIt _Last, boost::compute::command_queue &queue = boost::compute::system::default_queue()){
    boost::compute::iota(_First, _Last, 0, queue);
}

template <class _FwdIt>
void device_sequence_n(_FwdIt _First, size_t n, boost::compute::command_queue& queue = boost::compute::system::default_queue()){
    device_sequence(_First, _First + n, queue);
}

_KIAM_MATH_END
