#pragma once

#include <string>
#include <algorithm>
#include <numeric>
#include <numeric>
#include <tuple>

#include <boost/iterator/counting_iterator.hpp>

#include "type_traits.hpp"

_KIAM_MATH_BEGIN

template<typename T1, typename T2>
using math_pair = std::pair<T1, T2>;

template<typename T>
void math_swap(T &x, T &y) noexcept {
    std::swap(x, y);
}

template<typename T>
T math_abs(T const& x) noexcept {
    return std::abs(x);
}

template<class _FwdIt, class _Ty>
void math_fill(_FwdIt _First, _FwdIt _Last, const _Ty& _Val) noexcept {
    std::fill(_First, _Last, _Val);
}

template<class _FwdIt, class _Ty>
void math_fill_n(_FwdIt _First, size_t n, const _Ty& _Val) noexcept {
    std::fill_n(_First, n, _Val);
}

template<class _FwdIt, class _Ty>
void device_fill(_FwdIt _First, _FwdIt _Last, const _Ty& _Val){
    std::fill(EXECUTION_POLICY _First, _Last, _Val);
}

template<class _FwdIt, class _Ty>
void device_fill_n(_FwdIt _First, size_t n, const _Ty& _Val){
    std::fill_n(EXECUTION_POLICY _First, n, _Val);
}

template<class _InIt, class _OutIt>
_OutIt math_copy(_InIt _First, _InIt _Last, _OutIt _Dest) noexcept {
    return std::copy(_First, _Last, _Dest);
}

template<class _InIt, class _OutIt>
_OutIt math_copy_n(_InIt _First, size_t n, _OutIt _Dest) noexcept {
    return std::copy_n(_First, n, _Dest);
}

template<class _InIt, class _OutIt>
_OutIt device_copy(_InIt _First, _InIt _Last, _OutIt _Dest){
    return std::copy(EXECUTION_POLICY _First, _Last, _Dest);
}

template<class _InIt, class _OutIt>
_OutIt device_copy_n(_InIt _First, size_t n, _OutIt _Dest){
    return std::copy_n(EXECUTION_POLICY _First, n, _Dest);
}

template<class _InIt, class _OutIt>
_OutIt host_to_device_copy_n(_InIt _First, size_t n, _OutIt _Dest){
    return std::copy_n(EXECUTION_POLICY _First, n, _Dest);
}

template<class _InIt, class _OutIt>
_OutIt device_to_host_copy_n(_InIt _First, size_t n, _OutIt _Dest){
    return std::copy_n(EXECUTION_POLICY _First, n, _Dest);
}

template<class _InIt, class _Ty>
_Ty math_reduce(_InIt _First, _InIt _Last, _Ty _Val) noexcept {
#if _HAS_CXX17
    return std::reduce(_First, _Last, _Val);
#else
    return std::accumulate(_First, _Last, _Val);
#endif
}

template<class _InIt, class _Ty>
_Ty math_reduce_n(_InIt _First, size_t n, _Ty _Val) noexcept {
    return math_reduce(_First, _First + n, _Val);
}

template<class _InIt, class _Ty, class _BinOp>
_Ty math_reduce(_InIt _First, _InIt _Last, _Ty _Val, _BinOp _Reduce_op) noexcept {
#if _HAS_CXX17
    return std::reduce(_First, _Last, _Val, _Reduce_op);
#else
    return std::accumulate(_First, _Last, _Val, _Reduce_op);
#endif
}

template<class _InIt, class _Ty, class _BinOp>
_Ty math_reduce_n(_InIt _First, size_t n, _Ty _Val, _BinOp _Reduce_op) noexcept {
    return math_reduce(_First, _First + n, _Val, _Reduce_op);
}

template<class _InIt, class _Ty>
_Ty device_reduce(_InIt _First, _InIt _Last, _Ty _Val) noexcept {
#if _HAS_CXX17
    return std::reduce(EXECUTION_POLICY _First, _Last, _Val);
#else
    return std::accumulate(_First, _Last, _Val);
#endif
}

template<class _InIt, class _Ty>
_Ty device_reduce_n(_InIt _First, size_t n, _Ty _Val) noexcept {
    return device_reduce(_First, _First + n, _Val);
}

template<class _InIt, class _Ty, class _BinOp>
_Ty device_reduce(_InIt _First, _InIt _Last, _Ty _Val, _BinOp _Reduce_op) noexcept {
#if _HAS_CXX17
    return std::reduce(EXECUTION_POLICY _First, _Last, _Val, _Reduce_op);
#else
    return std::accumulate(_First, _Last, _Val, _Reduce_op);
#endif
}

template<class _InIt, class _Ty, class _BinOp>
_Ty device_reduce_n(_InIt _First, size_t n, _Ty _Val, _BinOp _Reduce_op) noexcept {
    return device_reduce(_First, _First + n, _Val, _Reduce_op);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt math_transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func) noexcept {
    return std::transform(_First, _Last, _Dest, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt math_transform_n(_InIt _First, size_t n, _OutIt _Dest, _Fn1 _Func) noexcept {
    return math_transform(_First, _First + n, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt math_transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func) noexcept {
    return std::transform(_First1, _Last1, _First2, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt math_transform_n(_InIt1 _First1, size_t n, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func) noexcept {
    return math_transform(_First1, _First1 + n, _First2, _Dest, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt device_transform(_InIt _First, _InIt _Last, _OutIt _Dest, _Fn1 _Func) noexcept {
    return std::transform(EXECUTION_POLICY _First, _Last, _Dest, _Func);
}

template<class _InIt, class _OutIt, class _Fn1>
_OutIt device_transform_n(_InIt _First, size_t n, _OutIt _Dest, _Fn1 _Func) noexcept {
    return device_transform(_First, _First + n, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt device_transform(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func) noexcept {
    return std::transform(EXECUTION_POLICY _First1, _Last1, _First2, _Dest, _Func);
}

template<class _InIt1, class _InIt2, class _OutIt, class _Fn2>
_OutIt device_transform_n(_InIt1 _First1, size_t n, _InIt2 _First2, _OutIt _Dest, _Fn2 _Func) noexcept {
    return device_transform(_First1, _First1 + n, _First2, _Dest, _Func);
}

template<class _FwdIt, class _Ty, class _BinOp, class _UnaryOp>
_Ty math_transform_reduce(_FwdIt _First, _FwdIt _Last, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op) noexcept {
#if _HAS_CXX17
    return std::transform_reduce(_First, _Last, _Val, _Reduce_op, _Transform_op);
#else
    for (; _First != _Last; ++_First)
        _Val = _Reduce_op(_Val, _Transform_op(*_First));
    return (_Val);
#endif
}

template<class _FwdIt, class _Ty, class _BinOp, class _UnaryOp>
_Ty math_transform_reduce_n(_FwdIt _First, size_t n, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op) noexcept {
    return math_transform_reduce(_First, _First + n, _Val, _Reduce_op, _Transform_op);
}

template<class _FwdIt1, class _FwdIt2, class _Ty, class _BinOp, class _UnaryOp>
_Ty math_transform_reduce(_FwdIt1 _First1, _FwdIt1 _Last1, _FwdIt2 _First2, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op) noexcept {
#if _HAS_CXX17
    return std::transform_reduce(_First1, _Last1, _First2, _Val, _Reduce_op, _Transform_op);
#else
    for (; _First1 != _Last1; ++_First1, ++_First2)
        _Val = _Reduce_op(_Val, _Transform_op(*_First1, *_First2));
    return (_Val);
#endif
}

template<class _FwdIt1, class _FwdIt2, class _Ty, class _BinOp, class _UnaryOp>
_Ty math_transform_reduce_n(_FwdIt1 _First1, size_t n, _FwdIt2 _First2, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op) noexcept {
    return math_transform_reduce(_First1, _First1 + n, _First2, _Val, _Reduce_op, _Transform_op);
}

template<class _FwdIt, class _Ty, class _BinOp, class _UnaryOp>
_Ty device_transform_reduce(_FwdIt _First, _FwdIt _Last, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op) noexcept {
#if _HAS_CXX17
    return std::transform_reduce(EXECUTION_POLICY _First, _Last, _Val, _Reduce_op, _Transform_op);
#else
    return math_transform_reduce(_First, _Last, _Val, _Reduce_op, _Transform_op);
#endif
}

template<class _FwdIt, class _Ty, class _BinOp, class _UnaryOp>
_Ty device_transform_reduce_n(_FwdIt _First, size_t n, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op) noexcept {
    return device_transform_reduce(_First, _First + n, _Val, _Reduce_op, _Transform_op);
}

template<class _FwdIt1, class _FwdIt2, class _Ty, class _BinOp, class _UnaryOp>
_Ty device_transform_reduce(_FwdIt1 _First1, _FwdIt1 _Last1, _FwdIt2 _First2, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op) noexcept {
#if _HAS_CXX17
    return std::transform_reduce(EXECUTION_POLICY _First1, _Last1, _First2, _Val, _Reduce_op, _Transform_op);
#else
    return math_transform_reduce(_First1, _Last1, _First2, _Val, _Reduce_op, _Transform_op);
#endif
}

template<class _FwdIt1, class _FwdIt2, class _Ty, class _BinOp, class _UnaryOp>
_Ty device_transform_reduce_n(_FwdIt1 _First1, size_t n, _FwdIt2 _First2, _Ty _Val, _BinOp _Reduce_op, _UnaryOp _Transform_op) noexcept {
    return device_transform_reduce(_First1, _First1 + n, _First2, _Val, _Reduce_op, _Transform_op);
}

template<class _InIt1, class _InIt2, class _Ty>
_Ty math_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val) noexcept {
    return std::inner_product(_First1, _Last1, _First2, _Val);
}

template<class _InIt1, class _InIt2, class _Ty>
_Ty math_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val) noexcept {
    return math_inner_product(_First1, _First1 + n, _First2, _Val);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
_Ty math_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2) noexcept {
    return std::inner_product(_First1, _Last1, _First2, _Val, _Func1, _Func2);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
_Ty math_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2) noexcept {
    return math_inner_product(_First1, _First1 + n, _First2, _Val, _Func1, _Func2);
}

template<class _InIt1, class _InIt2, class _Ty>
_Ty device_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val) noexcept {
    return std::inner_product(EXECUTION_POLICY _First1, _Last1, _First2, _Val);
}

template<class _InIt1, class _InIt2, class _Ty>
_Ty device_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val) noexcept {
    return device_inner_product(_First1, _First1 + n, _First2, _Val);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
_Ty device_inner_product(_InIt1 _First1, _InIt1 _Last1, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2) noexcept {
    return std::inner_product(EXECUTION_POLICY _First1, _Last1, _First2, _Val, _Func1, _Func2);
}

template<class _InIt1, class _InIt2, class _Ty, class _Fn21, class _Fn22>
_Ty device_inner_product_n(_InIt1 _First1, size_t n, _InIt2 _First2, _Ty _Val, _Fn21 _Func1, _Fn22 _Func2) noexcept {
    return device_inner_product(_First1, _First1 + n, _First2, _Val, _Func1, _Func2);
}

template<class _InIt, class _Fn1>
void math_for_each(_InIt _First, _InIt _Last, _Fn1 _Func) noexcept {
    std::for_each(_First, _Last, _Func);
}

template<class _InIt, class _Fn1>
void math_for_each_n(_InIt _First, size_t n, _Fn1 _Func) noexcept {
    math_for_each(_First, _First + n, _Func);
}

template<class _InIt, class _Fn1>
void device_for_each(_InIt _First, _InIt _Last, _Fn1 _Func) noexcept {
    std::for_each(EXECUTION_POLICY _First, _Last, _Func);
}

template<class _InIt, class _Fn1>
void device_for_each_n(_InIt _First, size_t n, _Fn1 _Func) noexcept {
    device_for_each(_First, _First + n, _Func);
}

template<class _FwdIt>
_FwdIt math_max_element(_FwdIt _First, _FwdIt _Last) noexcept {
    return std::max_element(_First, _Last);
}

template<class _FwdIt>
_FwdIt math_max_element_n(_FwdIt _First, size_t n) noexcept {
    return math_max_element(_First, _First + n);
}

template<class _FwdIt, class _Compare>
_FwdIt math_max_element(_FwdIt _First, _FwdIt _Last, _Compare _Comp) noexcept {
    return std::max_element(_First, _Last, _Comp);
}

template<class _FwdIt, class _Compare>
_FwdIt math_max_element_n(_FwdIt _First, size_t n, _Compare _Comp) noexcept {
    return math_max_element(_First, _First + n, _Comp);
}

template<class _FwdIt>
_FwdIt device_max_element(_FwdIt _First, _FwdIt _Last) noexcept {
    return std::max_element(EXECUTION_POLICY _First, _Last);
}

template<class _FwdIt>
_FwdIt device_max_element_n(_FwdIt _First, size_t n) noexcept {
    return device_max_element(_First, _First + n);
}

template<class _FwdIt, class _Compare>
_FwdIt device_max_element(_FwdIt _First, _FwdIt _Last, _Compare _Comp) noexcept {
    return std::max_element(EXECUTION_POLICY _First, _Last, _Comp);
}

template<class _FwdIt, class _Compare>
_FwdIt device_max_element_n(_FwdIt _First, size_t n, _Compare _Comp) noexcept {
    return device_max_element(_First, _First + n, _Comp);
}

template<class _FwdIt>
_FwdIt math_min_element(_FwdIt _First, _FwdIt _Last) noexcept {
    return std::min_element(_First, _Last);
}

template<class _FwdIt>
_FwdIt math_min_element_n(_FwdIt _First, size_t n) noexcept {
    return math_min_element(_First, _First + n);
}

template<class _FwdIt, class _Compare>
_FwdIt math_min_element(_FwdIt _First, _FwdIt _Last, _Compare _Comp) noexcept {
    return std::min_element(_First, _Last, _Comp);
}

template<class _FwdIt, class _Compare>
_FwdIt math_min_element_n(_FwdIt _First, size_t n, _Compare _Comp) noexcept {
    return math_min_element(_First, _First + n, _Comp);
}

template<class _FwdIt>
_FwdIt device_min_element(_FwdIt _First, _FwdIt _Last) noexcept {
    return std::min_element(EXECUTION_POLICY _First, _Last);
}

template<class _FwdIt>
_FwdIt device_min_element_n(_FwdIt _First, size_t n) noexcept {
    return device_min_element(_First, _First + n);
}

template<class _FwdIt, class _Compare>
_FwdIt device_min_element(_FwdIt _First, _FwdIt _Last, _Compare _Comp) noexcept {
    return std::min_element(EXECUTION_POLICY _First, _Last, _Comp);
}

template<class _FwdIt, class _Compare>
_FwdIt device_min_element_n(_FwdIt _First, size_t n, _Compare _Comp) noexcept {
    return device_min_element(_First, _First + n, _Comp);
}

template<class _InIt, class _Ty>
_InIt math_find(_InIt _First, const _InIt _Last, const _Ty& _Val){
    return std::find(_First, _Last, _Val);
}

template<class _InIt, class _Ty>
_InIt math_find_n(_InIt _First, size_t n, const _Ty& _Val){
    return math_find(_First, _First + n, _Val);
}

template<class _InIt, class _Ty>
_InIt device_find(_InIt _First, const _InIt _Last, const _Ty& _Val){
    return std::find(EXECUTION_POLICY _First, _Last, _Val);
}

template<class _InIt, class _Ty>
_InIt device_find_n(_InIt _First, size_t n, const _Ty& _Val){
    return device_find(_First, _First + n, _Val);
}

template<class _InIt, class _Pr>
_InIt math_find_if(_InIt _First, const _InIt _Last, _Pr _Pred){
    return std::find_if(_First, _Last, _Pred);
}

template<class _InIt, class _Pr>
_InIt math_find_if_n(_InIt _First, size_t n, _Pr _Pred){
    return math_find_if(_First, _First + n, _Pred);
}

template<class _InIt, class _Pr>
_InIt device_find_if(_InIt _First, const _InIt _Last, _Pr _Pred){
    return std::find_if(EXECUTION_POLICY _First, _Last, _Pred);
}

template<class _InIt, class _Pr>
_InIt device_find_if_n(_InIt _First, size_t n, _Pr _Pred){
    return device_find_if(_First, _First + n, _Pred);
}

template <class _FwdIt, class _Ty, class _Pr>
_FwdIt math_lower_bound(_FwdIt _First, const _FwdIt _Last, const _Ty& _Val, _Pr _Pred){
    return std::lower_bound(_First, _Last, _Val, _Pred);
}

template <class _FwdIt, class _Ty, class _Pr>
_FwdIt math_lower_bound_n(_FwdIt _First, size_t n, const _Ty& _Val, _Pr _Pred){
    return math_lower_bound(_First, _First + n, _Val, _Pred);
}

template <class _FwdIt, class _Ty>
_FwdIt math_lower_bound(_FwdIt _First, _FwdIt _Last, const _Ty& _Val){
    return std::lower_bound(_First, _Last, _Val);
}

template <class _FwdIt, class _Ty>
_FwdIt math_lower_bound_n(_FwdIt _First, size_t n, const _Ty& _Val){
    return math_lower_bound(_First, _First + n, _Val);
}

template <class _FwdIt, class _Ty, class _Pr>
_FwdIt math_upper_bound(_FwdIt _First, const _FwdIt _Last, const _Ty& _Val, _Pr _Pred){
    return std::upper_bound(_First, _Last, _Val, _Pred);
}

template <class _FwdIt, class _Ty, class _Pr>
_FwdIt math_upper_bound_n(_FwdIt _First, size_t n, const _Ty& _Val, _Pr _Pred){
    return math_upper_bound(_First, _First + n, _Val, _Pred);
}

template <class _FwdIt, class _Ty>
_FwdIt math_upper_bound(_FwdIt _First, _FwdIt _Last, const _Ty& _Val){
    return std::upper_bound(_First, _Last, _Val);
}

template <class _FwdIt, class _Ty>
_FwdIt math_upper_bound_n(_FwdIt _First, size_t n, const _Ty& _Val){
    return math_upper_bound(_First, _First + n, _Val);
}

template <class _FwdIt, class _Fn>
void math_generate(_FwdIt _First, _FwdIt _Last, _Fn _Func){
    std::generate(_First, _Last, _Func);
}

template <class _FwdIt, class _Fn>
void math_generate_n(_FwdIt _First, size_t n, _Fn _Func){
    math_generate(_First, _First + n, _Func);
}

template <class _FwdIt, class _Fn>
void device_generate(_FwdIt _First, _FwdIt _Last, _Fn _Func){
    std::generate(EXECUTION_POLICY _First, _Last, _Func);
}

template <class _FwdIt, class _Fn>
void device_generate_n(_FwdIt _First, size_t n, _Fn _Func){
    device_generate(_First, _First + n, _Func);
}

template <class _FwdIt>
void math_sequence(_FwdIt _First, _FwdIt _Last){
    std::iota(_First, _Last, 0);
}

template <class _FwdIt>
void math_sequence_n(_FwdIt _First, size_t n){
    math_sequence(_First, _First + n);
}

template <class _FwdIt>
void device_sequence(_FwdIt _First, _FwdIt _Last){
    std::iota(EXECUTION_POLICY _First, _Last, 0);
}

template <class _FwdIt>
void device_sequence_n(_FwdIt _First, size_t n){
    device_sequence(_First, _First + n);
}

template<typename F>
void execute_once(F f){
    f();
}

template<typename F, typename... Args>
void execute_n(size_t n, F f, Args... args){
    #pragma omp parallel for schedule(static)
    for(ssize_t i = 0; i < (ssize_t)n; ++i)
        f(i, args...);
}

using math_string = std::string;

template<class Incrementable>
using math_counting_iterator = boost::counting_iterator<Incrementable>;

template<typename... _Types>
using math_tuple = std::tuple<_Types...>;

template<typename... _Types>
auto make_math_tuple(_Types&&... args){
    return std::make_tuple(args...);
}

template<size_t _Index, class _Tuple>
using math_tuple_element = std::tuple_element<_Index, _Tuple>;

template<size_t _Index, class _Tuple>
using math_tuple_element_t = typename math_tuple_element<_Index, _Tuple>::type;

template<class _Tuple>
using math_tuple_size = std::tuple_size<_Tuple>;

template<size_t _Index, typename... _Types>
constexpr std::tuple_element_t<_Index, math_tuple<_Types...>>& math_get(math_tuple<_Types...> &t) noexcept {
    return std::get<_Index>(t);
}

template<size_t _Index, typename... _Types>
constexpr math_tuple_element_t<_Index, math_tuple<_Types...>>&& math_get(math_tuple<_Types...>&& t) noexcept {
    return std::get<_Index>(std::forward<math_tuple<_Types...>>(t));
}

template<size_t _Index, typename... _Types>
constexpr const math_tuple_element_t<_Index, math_tuple<_Types...>>& math_get(math_tuple<_Types...> const& t) noexcept {
    return std::get<_Index>(t);
}

template<typename T>
T* make_device_ptr(T* p){
    return p;
}

template<typename T>
const T* make_device_ptr(const T* p){
    return p;
}

template<typename T>
T* make_ptr(T* p){
    return p;
}

template<typename T>
const T* make_ptr(const T* p){
    return p;
}

_KIAM_MATH_END
