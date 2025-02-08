#pragma once

#include "func_traits.hpp"
#include "define_function.h"

_FUNCPROG_BEGIN

template<typename F>
struct is_function : std::false_type {};

template<typename FT>
struct is_function<function_t<FT> > : std::true_type{};

template<typename F>
bool constexpr is_function_v = is_function<F>::value;

template<typename T>
struct is_function0 : std::false_type {};

template<typename T>
struct is_function0<function_t<T()> > : std::true_type{};

template<typename F>
bool constexpr is_function0_v = is_function0<F>::value;

// Function composition

template<typename Ret, typename Arg, typename... Args>
function_t<Ret(Arg, Args...)> invoke_f0(function_t<Ret(Arg, Args...)> const& f) {
    return f;
}

template<typename Ret>
Ret invoke_f0(f0<Ret> const& f) {
    return f();
}

template<typename RetG, typename RetF, typename ArgF, typename... ArgsF, typename ArgG, typename... ArgsG>
std::enable_if_t<
    is_same_as_v<ArgG, function_t<RetF(ArgsF...)> >,
    function_t<RetG(ArgF, ArgsG...)>
> operator&(function_t<RetG(ArgG, ArgsG...)> const& g, function_t<RetF(ArgF, ArgsF...)> const& f){
    return [g, f](ArgF argF, ArgsG... argsG) { return g(invoke_f0(f << argF), argsG...); };
}

// Function application
template<typename Ret, typename Arg0, typename... Args>
function_t<Ret(Args...)> operator<<(
    function_t<Ret(Arg0, Args...)> const& f,
    fdecay<Arg0> const& arg0);

// Application to the function without parameters
template<typename Ret, typename Arg0, typename... Args>
function_t<Ret(Args...)> operator<<(
    function_t<Ret(Arg0, Args...)> const& f,
    f0<fdecay<Arg0> > const& arg0);

// Function dereference - same as func()
template<typename Ret>
Ret operator*(f0<Ret> const& f);

template<typename T>
T deref(f0<T> const&);

_FUNCPROG_END

namespace std {

template<typename Ret>
ostream& operator<<(ostream &os, _FUNCPROG::f0<Ret> const& f);

template<typename Ret>
wostream& operator<<(wostream &os, _FUNCPROG::f0<Ret> const& f);

template<typename T1, typename T2>
ostream& operator<<(ostream &os, pair<T1, T2> const& p);

template<typename T1, typename T2>
wostream& operator<<(wostream &os, pair<T1, T2> const& p);

template<typename T1, typename T2>
ostream& operator<<(ostream &os, _FUNCPROG::f0<pair<T1, T2> > const& f);

template<typename T1, typename T2>
wostream& operator<<(wostream &os, _FUNCPROG::f0<pair<T1, T2> > const& f);

template<typename T1, typename T2, typename T3>
ostream& operator<<(ostream &os, tuple<T1, T2, T3> const& t);

template<typename T1, typename T2, typename T3>
wostream& operator<<(wostream &os, tuple<T1, T2, T3> const& t);

template<typename T1, typename T2, typename T3>
ostream& operator<<(ostream &os, _FUNCPROG::f0<tuple<T1, T2, T3> > const& f);

template<typename T1, typename T2, typename T3>
wostream& operator<<(wostream &os, _FUNCPROG::f0<tuple<T1, T2, T3> > const& f);

} // namespace std
