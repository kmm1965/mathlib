#pragma once

#include "../funcprog_common.hpp"

_FUNCPROG_BEGIN

template<class _F>
struct _is_functor : std::false_type {};

template<class _F>
bool constexpr _is_functor_v = _is_functor<_F>::value;

//template<class F>
//using is_functor = _is_functor<base_class_t<F> >;
template<class _F>
struct is_functor : std::false_type {};

template<class F>
bool constexpr is_functor_v = is_functor<F>::value;

template<class F, typename T = F>
using functor_type = std::enable_if_t<is_functor_v<F>, T>;

#define FUNCTOR_TYPE_(F, T) BOOST_IDENTITY_TYPE((functor_type<F, T>))
#define FUNCTOR_TYPE(F, T) typename FUNCTOR_TYPE_(F, T)

template<class _F1, class _F2>
struct _is_same_functor : std::false_type {};

template<class _F>
struct _is_same_functor<_F, _F> : _is_functor<_F> {};

template<class _F1, class _F2>
bool constexpr _is_same_functor_v = _is_same_functor<_F1, _F2>::value;

template<class F1, class F2>
using is_same_functor = _is_same_functor<base_class_t<F1>, base_class_t<F2> >;

template<class F1, class F2>
bool constexpr is_same_functor_v = is_same_functor<F1, F2>::value;

template<class F1, class F2, typename T>
using same_functor_type = std::enable_if_t<is_same_functor<F1, F2>::value, T>;

#define SAME_FUNCTOR_TYPE_(F1, F2, T) BOOST_IDENTITY_TYPE((same_functor_type<F1, F2, T>))
#define SAME_FUNCTOR_TYPE(F1, F2, T) typename SAME_FUNCTOR_TYPE_(F1, F2, T)

// Requires operator/ (analogue to <$> in Haskell)
template<typename F>
struct Functor;

template<typename T>
using Functor_t = Functor<base_class_t<T> >;

template<typename T>
struct is_parser : std::false_type {};

template<typename T>
bool constexpr is_parser_v = is_parser<T>::value;

#define IMPLEMENT_FUNCTOR(_F, F) \
    template<> struct _is_functor<_F> : std::true_type {}; \
    template<typename A> struct is_functor<F<A> > : std::true_type {};

template<typename F, typename Ret, typename Arg, typename... Args>
using fmap_type = std::enable_if_t<
    is_functor_v<F> && !is_parser_v<F> && is_same_as_v<value_type_t<F>, Arg>,
    typeof_dt<F, function_t<Ret(Args...)> >
>;

#define FMAP_TYPE_(F, Ret, Arg) BOOST_IDENTITY_TYPE((fmap_type<F, Ret, Arg, Args...>))
#define FMAP_TYPE(F, Ret, Arg) typename FMAP_TYPE_(F, Ret, Arg)

//#define ENABLE_IF_NOT_PARSER_(F) BOOST_IDENTITY_TYPE((std::enable_if_t<!is_parser_v<F>, F>))
//#define ENABLE_IF_NOT_PARSER(F) typename ENABLE_IF_NOT_PARSER_(F)

// <$> fmap :: Functor f => (a -> b) -> f a -> f b
DECLARE_FUNCTION_2_ARGS(3, FMAP_TYPE(T0, T1, T2), fmap, function_t<T1(T2, Args...)> const&, T0 const&);

template<typename F, typename Ret, typename Arg, typename... Args>
constexpr auto operator/(function_t<Ret(Arg, Args...)> const& f, F const& v){
    return fmap(f, v);
}

// liftA == fmap
DECLARE_FUNCTION_2_ARGS(3, FMAP_TYPE(T0, T1, T2), liftA, function_t<T1(T2, Args...)> const&, T0 const&);

/*
-- | Replace all locations in the input with the same value.
--The default definition is @'fmap' . 'const'@, but this may be
-- overridden with a more efficient version.
(<$) :: a -> f b -> f a
*/
DECLARE_FUNCTION_2(2, FUNCTOR_TYPE(T0, TYPEOF_T(T0, T1)), left_fmap, T1 const&, T0 const&);

template<typename F, typename T>
constexpr auto operator/=(T const& v, F const& f){
    return left_fmap(v, f);
}

_FUNCPROG_END
