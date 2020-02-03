#pragma once

#include "funcprog_setup.h"

_FUNCPROG_BEGIN

template<typename Ret, typename... Args>
struct function_type_
{
    using type = function_t<Ret(Args...)>;
};

template<class F>  // Common case for functors & lambdas
struct function_type : function_type<decltype(&F::operator())> {};

template<typename Ret, class Cls, typename... Args>
struct function_type<Ret(Cls::*)(Args...)> : function_type_<Ret, Args...> {};

template<typename Ret, class Cls, typename... Args>
struct function_type<Ret(Cls::*)(Args...) const> : function_type_<Ret, Args...> {};

template<typename Ret, typename... Args>
struct function_type<Ret(*)(Args...)> : function_type_<Ret, Args...> {};

template<typename F>
using function_type_t = typename function_type<F>::type;

template<typename T>
using f0 = function_t<T()>;

template<typename T>
struct fdata
{
    using value_type = T;
    using func_type = f0<value_type>;

    fdata(value_type const& value) : func([value]() { return value; }) {}
    fdata(func_type const& func) : func(func) {}
    fdata(fdata const& other) : func(other.func) {}

    value_type operator()() const {
        return func();
    }

    operator value_type() const {
        return func();
    }

    value_type operator*() const {
        return func();
    }

    const func_type func;
};

template<typename T>
struct remove_fdata
{
    typedef T type;
};

template<typename T>
struct remove_fdata<fdata<T> >
{
    typedef T type;
};

template<typename T>
using remove_fdata_t = typename remove_fdata<T>::type;

template<typename T>
fdata<T> _fd(T const& v) {
    return v;
}

using _int = fdata<int>;
using _uint = fdata<unsigned int>;
using _double = fdata<double>;
using _string = fdata<std::string>;

template<typename T>
T value_of(T const& arg) {
    return arg;
}

template<typename T>
T value_of(f0<T> const& f) {
    return f();
}

template<typename T>
struct remove_f0
{
    typedef T type;
};

template<typename T>
struct remove_f0<f0<T> >
{
    typedef T type;
};

template<typename T>
using remove_f0_t = typename remove_f0<T>::type;

template<typename T>
using fdecay = remove_fdata_t<remove_f0_t<typename std::decay<T>::type > >;

template<typename FUNC>
struct first_argument_type;

template<typename FUNC>
using first_argument_type_t = typename first_argument_type<FUNC>::type;

template<typename Ret, typename Arg0, typename... Args>
struct first_argument_type<function_t<Ret(Arg0, Args...)> > {
    using type = fdecay<Arg0>;
};

template<typename FUNC>
struct remove_first_arg;

template<typename FUNC>
using remove_first_arg_t = typename remove_first_arg<FUNC>::type;

template<typename Ret, typename Arg0, typename... Args>
struct remove_first_arg<function_t<Ret(Arg0, Args...)> > {
    using type = function_t<Ret(Args...)>;
};

template<typename FUNC>
struct function_signature;

template<typename FUNC>
using function_signature_t = typename function_signature<FUNC>::type;

template<typename Ret, typename... Args>
struct function_signature<function_t<Ret(Args...)> > {
    using type = Ret(Args...);
};


template<typename F>
function_type_t<F> _(F f) {
    return f;
}

template<typename T>
f0<T> _f(T const& value) {
    return [value]() {
        return value;
    };
}

template<typename A, typename B>
using is_same_as = std::is_same<fdecay<A>, fdecay<B> >;

_FUNCPROG_END
