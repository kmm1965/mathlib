#pragma once

#include "fwd/Applicative_fwd.hpp"

_FUNCPROG_BEGIN

template<typename _AP>
struct _Applicative // Default implementation of some functions
{
    template<typename Ret, typename Arg1, typename Arg2, typename... Args>
    static constexpr function_t<typeof_dt<_AP, function_t<Ret(Args...)> >
        (typeof_dt<_AP, Arg1> const&, typeof_dt<_AP, Arg2> const&)>
    liftA2(function_t<Ret(Arg1, Arg2, Args...)> const& f);

    template<typename Ret, typename Arg1, typename Arg2, typename Arg3, typename... Args>
    static constexpr function_t<typeof_dt<_AP, function_t<Ret(Args...)> >
        (typeof_dt<_AP, Arg1> const&, typeof_dt<_AP, Arg2> const&, typeof_dt<_AP, Arg3> const&)>
    liftA3(function_t<Ret(Arg1, Arg2, Arg3, Args...)> const& f);
};

#define DECLARE_APPLICATIVE_CLASS(AP) \
    template<typename T> static constexpr AP<fdecay<T> > pure(T const& x); \
    template<typename Ret, typename Arg, typename... Args> \
    static constexpr AP<remove_f0_t<function_t<Ret(Args...)> > > \
    apply(AP<function_t<Ret(Arg, Args...)> > const& f, AP<fdecay<Arg> > const& v);

_FUNCPROG_END
