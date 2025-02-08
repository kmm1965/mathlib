#pragma once

#include "fwd/Functor_fwd.hpp"

_FUNCPROG_BEGIN

template<typename _F>
struct _Functor // Default implementation of some functions
{
    template<typename Ret, typename Arg, typename... Args>
    static constexpr auto liftA(function_t<Ret(Arg, Args...)> const& f){
        return _([f](typeof_dt<_F, Arg> const& v){
            return Functor<_F>::fmap(f, v);
        });
    }

    /*
    -- | Replace all locations in the input with the same value.
    --The default definition is @'fmap' . 'const'@, but this may be
    -- overridden with a more efficient version.
    (<$)        ::a->f b->f a
    */
    template<typename F, typename A>
    static constexpr typeof_t<_F, A> left_fmap(A const& v, F const& f);
};

#define DECLARE_FUNCTOR_CLASS(F) \
    /* <$> fmap :: Functor f => (a -> b) -> f a -> f b */ \
    template<typename Ret, typename Arg, typename... Args> \
    static constexpr F<remove_f0_t<function_t<Ret(Args...)> > > \
    fmap(function_t<Ret(Arg, Args...)> const& f, F<fdecay<Arg> > const& v);

_FUNCPROG_END
