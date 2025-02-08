#pragma once

#include "func2_traits.hpp"
#include "define_function2.h"

_FUNCPROG2_BEGIN

template<typename F>
struct is_function : std::false_type {};

template<typename FuncType, typename FuncImpl>
struct is_function<function2<FuncType, FuncImpl> > : std::true_type{};

template<typename F>
bool constexpr is_function_v = is_function<F>::value;

template<typename T>
struct is_function0 : std::false_type {};

template<typename T, typename FuncImpl>
struct is_function0<function2<T(), FuncImpl> > : std::true_type{};

template<typename F>
bool constexpr is_function0_v = is_function0<F>::value;

template<typename Ret, typename Arg0, typename... Args, typename FuncImpl>
__DEVICE constexpr auto invoke_f0(function2<Ret(Arg0, Args...), FuncImpl> const& f){
    return f;
}

template<typename Ret, typename FuncImpl>
__DEVICE auto invoke_f0(f0<Ret, FuncImpl> const& f){
    return f();
}

template<typename FuncImpl, typename Ret, typename Arg0, typename... Args>
struct function_application
{
    using func_type = function2<Ret(Arg0, Args...), FuncImpl>;

    __DEVICE function_application(func_type const& f, Arg0 arg0) : f(f), arg0(arg0){}

    __DEVICE constexpr Ret operator()(Args... args) const {
        return f(arg0, args...);
    }

private:
    func_type const f;
    Arg0 arg0;
};

// Function application
template<typename Ret, typename Arg0, typename... Args, typename FuncImpl>
__DEVICE constexpr auto operator<<(function2<Ret(Arg0 const&, Args...), FuncImpl> const& f, Arg0 const& arg0){
    return _(function_application<FuncImpl, Ret, Arg0, Args...>(f, arg0));
}

template<typename Ret, typename Arg0, typename... Args, typename FuncImpl>
__DEVICE constexpr auto apply_arg(function2<Ret(Arg0, Args...), FuncImpl> const& f, Arg0 arg0){
    return _(function_application<FuncImpl, Ret, Arg0, Args...>(f, arg0));
}

// Function dereference - same as func()
template<typename Ret, typename FuncImpl>
__DEVICE constexpr Ret operator*(f0<Ret, FuncImpl> const& f){
    return f();
}

template<typename Ret, typename FuncImpl>
__DEVICE constexpr Ret deref(f0<Ret, FuncImpl> const& f){
    return f();
}

// Function composition
template<typename RetG, typename RetF, typename ArgF, typename... ArgsF, typename FuncImplF, typename ArgG, typename... ArgsG, typename FuncImplG>
constexpr auto operator&(function2<RetG(ArgG, ArgsG...), FuncImplG> const& g, function2<RetF(ArgF, ArgsF...), FuncImplF> const& f)
{
    static_assert(is_same_as_v<ArgG, function2<RetF(ArgsF...), FuncImplF> >, "The first argument type of g must equal to the result type of f");
    return _([g, f](ArgF argF, ArgsG... argsG){ return g(invoke_f0(f << argF), argsG...); });
}

_FUNCPROG2_END

namespace std {

    template<typename Ret, typename FuncImpl>
    ostream& operator<<(ostream& os, _FUNCPROG2::f0<Ret, FuncImpl> const& f){
        return os << *f;
    }

    template<typename Ret, typename FuncImpl>
    wostream& operator<<(wostream& os, _FUNCPROG2::f0<Ret, FuncImpl> const& f){
        return os << *f;
    }

} // namespace std
