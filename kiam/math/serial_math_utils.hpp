#pragma once

#include <string>
#include <algorithm>
#include <numeric>
#include <utility>  // pair

#include "math_def.h"

_KIAM_MATH_BEGIN

template<typename T1, typename T2>
#ifdef DONT_USE_CXX_11
struct math_pair : std::pair<T1, T2>{};
#else
using math_pair = std::pair<T1, T2>;
#endif

template<typename T>
void math_swap(T &x, T &y){
    std::swap(x, y);
}

template<class _FwdIt, class _Ty>
void math_fill(_FwdIt _First, _FwdIt _Last, const _Ty& _Val){
    std::fill(_First, _Last, _Val);
}

template<class _FwdIt, class _Ty>
void math_fill_n(_FwdIt _First, size_t n, const _Ty& _Val){
    std::fill_n(_First, n, _Val);
}

template<class _InIt, class _OutIt>
_OutIt math_copy(_InIt _First, _InIt _Last, _OutIt _Dest){
    return std::copy(_First, _Last, _Dest);
}

template<class _InIt, class _OutIt>
_OutIt math_copy_n(_InIt _First, size_t n, _OutIt _Dest){
    return std::copy(_First, _First + n, _Dest);
}

template<class _InIt, class _Ty>
_Ty math_accumulate(_InIt _First, _InIt _Last, _Ty _Val){
    return std::accumulate(_First, _Last, _Val);
}

template<class _InIt, class _Ty>
_Ty math_accumulate_n(_InIt _First, size_t n, _Ty _Val){
    return std::accumulate(_First, _First + n, _Val);
}

template<class _InIt, class _Ty, class _Fn2>
_Ty math_accumulate(_InIt _First, _InIt _Last, _Ty _Val, _Fn2 _Func){
    return std::accumulate(_First, _Last, _Val, _Func);
}

template<class _InIt, class _Ty, class _Fn2>
_Ty math_accumulate_n(_InIt _First, size_t n, _Ty _Val, _Fn2 _Func){
    return std::accumulate(_First, _First + n, _Val, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt math_transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func){
    return std::transform(_First, _Last, _Dest, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt math_transform_n(_InIt _First, size_t n, _OutIt _Dest, _Fn1 _Func){
    return std::transform(_First, _First + n, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt math_transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func){
    return std::transform(_First1, _Last1, _First2, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt math_transform_n(_InIt1 _First1, size_t n, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func){
    return std::transform(_First1, _First1 + n, _First2, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
_Ty math_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2){
    return std::inner_product(_First1, _Last1, _First2, _Val, _Func1, _Func2);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
_Ty math_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2){
    return std::inner_product(_First1, _First1 + n, _First2, _Val, _Func1, _Func2);
}

template<class _InIt1, class _InIt2, class _Ty>
_Ty math_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val){
    return std::inner_product(_First1, _Last1, _First2, _Val);
}

template<class _InIt1, class _InIt2, class _Ty>
_Ty math_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val){
    return std::inner_product(_First1, _First1 + n, _First2, _Val);
}

template<class _InIt, class _Fn1>
void math_for_each(_InIt _First, _InIt _Last, _Fn1 _Func){
    std::for_each(_First, _Last, _Func);
}

template<class _InIt, class _Fn1>
void math_for_each_n(_InIt _First, size_t n, _Fn1 _Func){
    math_for_each(_First, _First + n, _Func);
}

using math_string = std::string;

_KIAM_MATH_END
