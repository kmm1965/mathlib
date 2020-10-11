#pragma once

#include "Functor.hpp"

_FUNCPROG_BEGIN

template<class _A>
struct _is_applicative : std::false_type {};

template<class A>
using is_applicative = _is_applicative<base_class_t<A> >;

template<class A, typename T = A>
using applicative_type = typename std::enable_if<is_applicative<A>::value, T>::type;

template<class _A1, class _A2>
struct _is_same_applicative : std::false_type {};

template<class _A>
struct _is_same_applicative<_A, _A> : _is_applicative<_A> {};

template<class A1, class A2>
using is_same_applicative = _is_same_applicative<base_class_t<A1>, base_class_t<A2> >;

template<class A1, class A2, typename T>
using same_applicative_type = typename std::enable_if<is_same_applicative<A1, A2>::value, T>::type;

// Requires pure, / (analogue to <$> in Haskell) and * (analogue to <*> in Haskell)
template<typename AP>
struct Applicative;

template<typename T>
using Applicative_t = Applicative<base_class_t<T> >;

#define IMPLEMENT_APPLICATIVE(_AP) \
    template<> struct _is_applicative<_AP> : std::true_type {}

#define DECLARE_APPLICATIVE_CLASS(AP) \
    template<typename T> static constexpr AP<fdecay<T> > pure(T const& x); \
    template<typename Ret, typename Arg, typename... Args> \
    static constexpr AP<remove_f0_t<function_t<Ret(Args...)> > > \
    apply(AP<function_t<Ret(Arg, Args...)> > const& f, AP<fdecay<Arg> > const& v);


template<class AP, class AF>
struct apply_result_type
{
    static_assert(is_same_applicative<AP, AF>::value, "Should be the same Applicative");
    static_assert(is_function<value_type_t<AF> >::value, "Should be a function");
    static_assert(std::is_same<value_type_t<AP>, first_argument_type_t<value_type_t<AF> > >::value, "Should be the same");
    
    using type = typename AP::template type<remove_f0_t<remove_first_arg_t<value_type_t<AF> > > >;
};

template<class AP, class AF>
using apply_result_type_t = typename apply_result_type<AP, AF>::type;

// <*> :: Applicative f => f (a -> b) -> f a -> f b
template<class AP, class AF>
using apply_type = typename std::enable_if<
    is_same_applicative<AP, AF>::value &&
    is_function<value_type_t<AF> >::value &&
    std::is_same<value_type_t<AP>, first_argument_type_t<value_type_t<AF> > >::value,
    apply_result_type_t<AP, AF>
>::type;

#define _APPLY_TYPE(AP, AF) BOOST_IDENTITY_TYPE((apply_type<AP, AF>))
#define APPLY_TYPE(AP, AF) typename _APPLY_TYPE(AP, AF)

DEFINE_FUNCTION_2(2, constexpr APPLY_TYPE(T0, T1), apply, T1 const&, f, T0 const&, v,
    return Applicative_t<T0>::apply(f, v);)

template<class AP, class AF>
constexpr apply_type<AP, AF> operator*(AF const& f, AP const& v){
    return apply(f, v);
}

// liftA2 :: (a -> b -> c) -> f a -> f b -> f c
template<class AY, class AX, typename FT>
using liftA2_type = typename std::enable_if<
    is_same_applicative<AX, AY>::value &&
    std::is_same<value_type_t<AX>, first_argument_type_t<function_t<FT> > >::value &&
    std::is_same<value_type_t<AY>, first_argument_type_t<remove_first_arg_t<function_t<FT> > > >::value,
    apply_result_type_t<AY, fmap_result_type_t<AX, function_t<FT> > >
>::type;

#define LIFTA2_TYPE_(AY, AX, FT) BOOST_IDENTITY_TYPE((liftA2_type<AY, AX, FT>))
#define LIFTA2_TYPE(AY, AX, FT) typename LIFTA2_TYPE_(AY, AX, FT)

DEFINE_FUNCTION_3(3, constexpr LIFTA2_TYPE(T0, T1, T2), liftA2, function_t<T2> const&, f, T1 const&, x, T0 const&, y,
    return f / x * y;)

// liftA3 :: Applicative f => (a -> b -> c -> d) -> f a -> f b -> f c -> f d
template<class AZ, class AY, class AX, typename FT>
using liftA3_type = typename std::enable_if<
    is_same_applicative<AX, AY>::value &&
    is_same_applicative<AX, AZ>::value &&
    std::is_same<value_type_t<AX>, first_argument_type_t<function_t<FT> > >::value &&
    std::is_same<value_type_t<AY>, first_argument_type_t<remove_first_arg_t<function_t<FT> > > >::value &&
    std::is_same<value_type_t<AZ>, first_argument_type_t<remove_first_arg_t<remove_first_arg_t<function_t<FT> > > > >::value,
    apply_result_type_t<AZ, apply_result_type_t<AY, fmap_result_type_t<AX, function_t<FT> > > >
>::type;

#define LIFTA3_TYPE_(AZ, AY, AX, FT) BOOST_IDENTITY_TYPE((liftA3_type<AZ, AY, AX, FT>))
#define LIFTA3_TYPE(AZ, AY, AX, FT) typename LIFTA3_TYPE_(AZ, AY, AX, FT)

DEFINE_FUNCTION_4(4, constexpr LIFTA3_TYPE(T0, T1, T2, T3), liftA3, function_t<T3> const&, f, T2 const&, x, T1 const&, y, T0 const&, z,
    return f / x * y * z;)

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

template<typename Fa, typename Fb>
constexpr same_applicative_type<Fa, Fb, Fb> operator*=(Fa const& a, Fb const& b){
    return ap_r(a, b);
}

// (<*) = liftA2 const
template<typename Fa, typename Fb>
constexpr same_applicative_type<Fa, Fb, Fa> ap_l(Fa const& a, Fb const& b){
    return liftA2(_(const_<typename Fb::value_type, typename Fa::value_type>), a, b);
}

template<typename Fa, typename Fb>
constexpr same_applicative_type<Fa, Fb, Fa> operator^=(Fa const& a, Fb const& b){
    return ap_l(a, b);
}

_FUNCPROG_END
