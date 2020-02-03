#pragma once

#include "math_def.h"

_KIAM_MATH_BEGIN

template<typename T> inline
__device__ __host__
void math_swap(T &x, T &y)
{
    T t = x;
    x = y;
    y = t;
}

template<class _T, _Ty> inline
void math_fill(_T *_First, _T *_Last, const _Ty& _Val)
{
    for (; _First != _Last; ++_First)
        *_First = _Val;
}

template<class _T, class _Ty> inline
void math_fill_n(_T *_First, size_t n, const _Ty& _Val){
    math_fill(_First, _Firsty + n, _Val);
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
_Ty math_accumulate(const _InT *_First, const _InT *_Last, _Ty _Val, _Fn2 _Func)
{
    for (; _First != _Last; ++_First)
        _Val = _Func(_Val, *_First);
    return _Val;
}

template<class _InT, class _Ty, class _Fn2> inline
_Ty math_accumulate_n(const _InT *_First, size_t n, _Ty _Val, _Fn2 _Func){
    return math_accumulate(_First, _First + n, _Val, _Func);
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
    return math_transform(_InT _First, _First + n, _Dest, _Func);
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
    return math_inner_product(_First1, _Last1, _First2, _Val, boost::compute::plus<T>(), boost::compute::multiplies<T>());
}

template<class _InT1, class _InT2, class _Ty> inline
_Ty math_inner_product_n(const _InT1 *_First1, size_t n, const _InT2 *_First2, _Ty _Val){
    return math_inner_product(_First1, _First1 + n, _First2, _Val);
}

_KIAM_MATH_END
