#pragma once

#include "../Applicative.hpp"
#include "../Functor.hpp"

_FUNCPROG_BEGIN

template<typename _AP>
template<typename Ret, typename Arg1, typename Arg2, typename... Args>
constexpr function_t<typeof_dt<_AP, function_t<Ret(Args...)> >
    (typeof_dt<_AP, Arg1> const&, typeof_dt<_AP, Arg2> const&)>
_Applicative<_AP>::liftA2(function_t<Ret(Arg1, Arg2, Args...)> const& f){
    return _([f](typeof_dt<_AP, Arg1> const& a1, typeof_dt<_AP, Arg2> const& a2){
        using Appl = Applicative<_AP>;
        return Appl::apply(Functor<_AP>::fmap(f, a1), a2);
        //return Appl::apply(Appl::apply(Appl::pure(f), a1), a2);
    });
}

template<typename _AP>
template<typename Ret, typename Arg1, typename Arg2, typename Arg3, typename... Args>
constexpr function_t<typeof_dt<_AP, function_t<Ret(Args...)> >
    (typeof_dt<_AP, Arg1> const&, typeof_dt<_AP, Arg2> const&, typeof_dt<_AP, Arg3> const&)>
_Applicative<_AP>::liftA3(function_t<Ret(Arg1, Arg2, Arg3, Args...)> const& f){
    return _([f](typeof_dt<_AP, Arg1> const& a1, typeof_dt<_AP, Arg2> const& a2, typeof_dt<_AP, Arg3> const& a3){
        using Appl = Applicative<_AP>;
        return Appl::apply(Appl::apply(Functor<_AP>::fmap(f, a1), a2), a3);
        //return Appl::apply(Appl::apply(Appl::apply(Appl::pure(f), a1), a2), a3);
    });
}

FUNCTION_TEMPLATE(1) constexpr apply_type<T0> apply(T0 const& f, apply_arg_type<T0> const& v){
    return Applicative_t<T0>::apply(f, v);
}

// liftA2 :: (a -> b -> c) -> f a -> f b -> f c
FUNCTION_TEMPLATE_ARGS(5) constexpr LIFTA2_TYPE(T0, T1, T2, T3, T4) liftA2(function_t<T2(T3, T4, Args...)> const& f, T1 const& x, T0 const& y) {
    return Applicative_t<T0>::liftA2(f)(x, y);
}

// liftA3 :: Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
FUNCTION_TEMPLATE_ARGS(7) constexpr LIFTA3_TYPE(T0, T1, T2, T3, T4, T5, T6) liftA3(function_t<T3(T4, T5, T6, Args...)> const& f, T2 const& x, T1 const& y, T0 const& z) {
    return Applicative_t<T0>::liftA3(f)(x, y, z);
}

/*
-- | Sequence actions, discarding the value of the first argument.
(*>) ::f a->f b->f b
a1 *> a2 = (id <$ a1) <*> a2
-- This is essentially the same as liftA2(flip const), but if the
-- Functor instance has an optimized(<$), it may be better to use
-- that instead.Before liftA2 became a method, this definition
-- was strictly better, but now it depends on the Functor.For a
-- Functor supporting a sharing - enhancing(<$), this definition
-- may reduce allocation by preventing a1 from ever being fully
-- realized.In an implementation with a boring(<$) but an optimizing
-- liftA2, it would likely be better to define(*>) using liftA2.

-- | Sequence actions, discarding the value of the second argument.
(<*) ::f a->f b->f a
(<*) = liftA2 const
*/
// (*>) a1 *> a2 = (id <$ a1) <*> a2
template<typename Fa, typename Fb>
constexpr same_applicative_type<Fa, Fb, Fb> ap_r(Fa const& a, Fb const& b){
    return (_(id<typename Fb::value_type>) /= a) * b;
}

// (<*) = liftA2 const
template<typename Fa, typename Fb>
constexpr same_applicative_type<Fa, Fb, Fa> ap_l(Fa const& a, Fb const& b){
    return liftA2(_(const_<typename Fb::value_type, typename Fa::value_type>), a, b);
}

_FUNCPROG_END
