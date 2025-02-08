#pragma once

#include "Functor_fwd.hpp"

_FUNCPROG2_BEGIN

template<class _A>
struct _is_applicative : std::false_type {};

template<class _A>
bool constexpr _is_applicative_v = _is_applicative<_A>::value;

//template<class A>
//using is_applicative = _is_applicative<base_class_t<A> >;
template<class A>
struct is_applicative : std::false_type {};

template<class A>
bool constexpr is_applicative_v = is_applicative<A>::value;

template<class A, typename T = A>
using applicative_type = std::enable_if_t<is_applicative<A>::value, T>;

#define APPLICATIVE_TYPE_(A, T) BOOST_IDENTITY_TYPE((applicative_type<A, T>))
#define APPLICATIVE_TYPE(A, T) typename APPLICATIVE_TYPE_(A, T)

template<class _A1, class _A2>
struct _is_same_applicative : std::false_type {};

template<class _A>
struct _is_same_applicative<_A, _A> : _is_applicative<_A> {};

template<class _A1, class _A2>
bool constexpr _is_same_applicative_v = _is_same_applicative<_A1, _A2>::value;

template<class A1, class A2>
using is_same_applicative = _is_same_applicative<base_class_t<A1>, base_class_t<A2> >;

template<class A1, class A2>
bool constexpr is_same_applicative_v = is_same_applicative<A1, A2>::value;

template<class A1, class A2, typename T>
using same_applicative_type = std::enable_if_t<is_same_applicative<A1, A2>::value, T>;

#define SAME_APPLICATIVE_TYPE_(A1, A2, T) BOOST_IDENTITY_TYPE((same_applicative_type<A1, A2, T>))
#define SAME_APPLICATIVE_TYPE(A1, A2, T) typename SAME_APPLICATIVE_TYPE_(A1, A2, T)

// Requires pure, / (analogue to <$> in Haskell) and * (analogue to <*> in Haskell)
template<typename _AP>
struct Applicative;

template<typename T>
using Applicative_t = Applicative<base_class_t<T> >;

#define IMPLEMENT_APPLICATIVE(_AP, AP) \
    template<> struct _is_applicative<_AP> : std::true_type {}; \
    template<typename A> struct is_applicative<AP<A> > : std::true_type {};

// <*> :: Applicative f => f (a -> b) -> f a -> f b
template<class AF>
using apply_type = std::enable_if_t<
    is_applicative_v<AF> && is_function_v<value_type_t<AF> >,
    typeof_t<AF, remove_f0_t<remove_first_arg_t<value_type_t<AF> > > >
>;

template<class AF>
using apply_arg_type = typeof_dt<AF, first_argument_type_t<value_type_t<AF> > >;

//DECLARE_FUNCTION_2(1, apply_type<T0>, apply, T0 const&, apply_arg_type<T0> const&);

//template<class F, class AF>
//constexpr auto operator*(AF const& f, F const& v) {
//    return apply(f, v);
//}

//template<class AF>
//constexpr apply_type<AF> operator*(AF const& f, apply_arg_type<AF> const& v);

// liftA2 :: (a -> b -> c) -> f a -> f b -> f c
template<class AY, class AX, typename Ret, typename Arg1, typename Arg2, typename... Args>
using liftA2_type = std::enable_if_t<
    is_same_applicative_v<AX, AY> &&
    is_same_as_v<value_type_t<AX>, Arg1> &&
    is_same_as_v<value_type_t<AY>, Arg2>,
    typeof_t<AY, remove_f0_t<function2<Ret(Args...), void> > >
>;

#define LIFTA2_TYPE_(AY, AX, Ret, Arg1, Arg2) BOOST_IDENTITY_TYPE((liftA2_type<AY, AX, Ret, Arg1, Arg2, Args...>))
#define LIFTA2_TYPE(AY, AX, Ret, Arg1, Arg2) typename LIFTA2_TYPE_(AY, AX, Ret, Arg1, Arg2)

DECLARE_FUNCTION_3_ARGS(6, LIFTA2_TYPE(T0, T1, T3, T4, T5), liftA2, FUNCTION2(T3(T4, T5, Args...), T2) const&, T1 const&, T0 const&);

// liftA3 :: Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
template<class AZ, class AY, class AX, typename Ret, typename Arg1, typename Arg2, typename Arg3, typename... Args>
using liftA3_type = std::enable_if_t<
    is_same_applicative_v<AX, AY> &&
    is_same_applicative_v<AX, AZ> &&
    is_same_as_v<value_type_t<AX>, Arg1> &&
    is_same_as_v<value_type_t<AY>, Arg2> &&
    is_same_as_v<value_type_t<AZ>, Arg3>,
    typeof_t<AZ, remove_f0_t<function2<Ret(Args...), void> > >
>;

#define LIFTA3_TYPE_(AZ, AY, AX, Ret, Arg1, Arg2, Arg3) BOOST_IDENTITY_TYPE((liftA3_type<AZ, AY, AX, Ret, Arg1, Arg2, Arg3, Args...>))
#define LIFTA3_TYPE(AZ, AY, AX, Ret, Arg1, Arg2, Arg3) typename LIFTA3_TYPE_(AZ, AY, AX, Ret, Arg1, Arg2, Arg3)

DECLARE_FUNCTION_4_ARGS(8, LIFTA3_TYPE(T0, T1, T2, T4, T5, T6, T7), liftA3, FUNCTION2(T4(T5, T6, T7, Args...), T3) const&, T2 const&, T1 const&, T0 const&);

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
constexpr same_applicative_type<Fa, Fb, Fb> ap_r(Fa const& a, Fb const& b);

template<typename Fa, typename Fb>
constexpr same_applicative_type<Fa, Fb, Fb> operator*=(Fa const& a, Fb const& b){
    return ap_r(a, b);
}

// (<*) = liftA2 const
template<typename Fa, typename Fb>
constexpr same_applicative_type<Fa, Fb, Fa> ap_l(Fa const& a, Fb const& b);

template<typename Fa, typename Fb>
constexpr same_applicative_type<Fa, Fb, Fa> operator^=(Fa const& a, Fb const& b){
    return ap_l(a, b);
}

_FUNCPROG2_END
