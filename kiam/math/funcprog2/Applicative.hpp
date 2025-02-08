#pragma once

#include "fwd/Applicative_fwd.hpp"

_FUNCPROG2_BEGIN

template<typename _AP>
struct _Applicative // Default implementation of some functions
{
    template<typename Ret, typename Arg1, typename Arg2, typename... Args, typename FuncImpl>
    static constexpr function2<typeof_dt<_AP, function2<Ret(Args...), void> >
        (typeof_dt<_AP, Arg1> const&, typeof_dt<_AP, Arg2> const&), void>
    liftA2(function2<Ret(Arg1, Arg2, Args...), FuncImpl> const& f);

    template<typename Ret, typename Arg1, typename Arg2, typename Arg3, typename... Args, typename FuncImpl>
    static constexpr function2<typeof_dt<_AP, function2<Ret(Args...), void> >
        (typeof_dt<_AP, Arg1> const&, typeof_dt<_AP, Arg2> const&, typeof_dt<_AP, Arg3> const&), void>
    liftA3(function2<Ret(Arg1, Arg2, Arg3, Args...), FuncImpl> const& f);
};

#define DECLARE_APPLICATIVE_CLASS(AP) \
    template<typename T> static constexpr AP<fdecay<T> > pure(T const& x); \
    template<typename Ret, typename Arg, typename... Args> \
    static constexpr AP<remove_f0_t<function2<Ret(Args...)> > > \
    apply(AP<function2<Ret(Arg, Args...)> > const& f, AP<fdecay<Arg> > const& v);

_FUNCPROG2_END
