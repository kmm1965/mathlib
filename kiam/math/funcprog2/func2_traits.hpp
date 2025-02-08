#pragma once

#include "function2.hpp"

_FUNCPROG2_BEGIN

template<typename FuncImpl, typename Ret, typename... Args>
struct function2_type_
{
    using type = function2<Ret(Args...), FuncImpl>;
};

template<class F>  // Common case for functors & lambdas
struct function2_type;

template<class F>  // Common case for functors & lambdas
struct function2_type : function2_type<decltype(&F::operator())>{};

template<class Cls, typename Ret, typename... Args>
struct function2_type<Ret(Cls::*)(Args...)> : function2_type_<Cls, Ret, Args...>{};

template<class Cls, typename Ret, typename... Args>
struct function2_type<Ret(Cls::*)(Args...) const> : function2_type_<Cls, Ret, Args...>{};

template<typename F>
using function2_type_t = typename function2_type<F>::type;

template<typename T, typename FuncImpl>
using f0 = function2<T(), FuncImpl>;

template<typename T>
__DEVICE
constexpr T value_of(T const& arg){
    return arg;
}

template<typename T, typename FuncImpl>
__DEVICE
constexpr T value_of(f0<T, FuncImpl> const& f){
    return f();
}

template<typename T>
struct remove_f0
{
    typedef T type;
};

template<typename T, typename FuncImpl>
struct remove_f0<f0<T, FuncImpl> >
{
    typedef T type;
};

template<typename T>
using remove_f0_t = typename remove_f0<T>::type;

template<typename T>
using fdecay = remove_f0_t<typename std::decay<T>::type>;

template<typename FUNC>
struct result_type;

template<typename FuncImpl, typename Ret, typename... Args>
struct result_type<function2<Ret(Args...), FuncImpl> > {
    using type = Ret;
};

template<typename FUNC>
using result_type_t = typename result_type<FUNC>::type;

template<typename FUNC>
struct first_argument_type;

template<typename FuncImpl, typename Ret, typename Arg0, typename... Args>
struct first_argument_type<function2<Ret(Arg0, Args...), FuncImpl> >{
    using type = Arg0;
};

template<typename FUNC>
using first_argument_type_t = typename first_argument_type<FUNC>::type;

template<typename FUNC>
struct remove_first_arg;

template<typename FuncImpl, typename Ret, typename Arg0, typename... Args>
struct function_application;

template<typename FuncImpl, typename Ret, typename Arg0, typename... Args>
struct remove_first_arg<function2<Ret(Arg0, Args...), FuncImpl> >{
    using type = function2<Ret(Args...), function_application<FuncImpl, Ret, Arg0, Args...> >;
};

template<typename FUNC>
using remove_first_arg_t = typename remove_first_arg<FUNC>::type;

template<typename FUNC>
struct function_signature;

template<typename FUNC>
using function_signature_t = typename function_signature<FUNC>::type;

template<typename FuncImpl, typename Ret, typename... Args>
struct function_signature<function2<Ret(Args...), FuncImpl> >{
    using type = Ret(Args...);
};

template<typename F>
__DEVICE __HOST constexpr function2_type_t<F> _(F f){
    return f;
}

//template<typename T>
//__DEVICE constexpr auto _f(T const& value){
//    return _([value](){
//        return value;
//    });
//}

template<typename A>
using value_type_t = typename A::value_type;

template<typename A, typename T>
using typeof_t = typename A::template type<T>;

#define TYPEOF_T(A, T) typename A::template type<T>

template<typename A, typename T>
using typeof_dt = typeof_t<A, fdecay<T> >;

template<typename F>
using base_class_t = typename F::base_class;

template<typename A, typename B>
using is_same_as = std::is_same<fdecay<A>, fdecay<B> >;

template<typename A, typename B>
bool constexpr is_same_as_v = is_same_as<A, B>::value;

_FUNCPROG2_END
