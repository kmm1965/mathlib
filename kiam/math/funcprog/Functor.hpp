#pragma once

#include "funcprog_common.hpp"

_FUNCPROG_BEGIN

template<typename A>
using value_type_t = typename A::value_type;

template<typename A, typename T>
using typeof_t = typename A::template type<T>;

template<typename F>
using base_class_t = typename F::base_class;

template<class _F>
struct _is_functor : std::false_type {};

template<class F>
using is_functor = _is_functor<base_class_t<F> >;

template<class F, typename T = F>
using functor_type = typename std::enable_if<is_functor<F>::value, T>::type;

template<class _F1, class _F2>
struct _is_same_functor : std::false_type {};

template<class _F>
struct _is_same_functor<_F, _F> : _is_functor<_F> {};

template<class F1, class F2>
using is_same_functor = _is_same_functor<base_class_t<F1>, base_class_t<F2> >;

template<class F1, class F2, typename T>
using same_functor_type = typename std::enable_if<is_same_functor<F1, F2>::value, T>::type;

// Requires operator/ (analogue to <$> in Haskell)
template<typename F>
struct Functor;

template<typename T>
using Functor_t = Functor<base_class_t<T> >;

#define IMPLEMENT_FUNCTOR(_F) \
    template<> struct _is_functor<_F> : std::true_type {}

#define DECLARE_FUNCTOR_CLASS(F) \
    /* <$> fmap :: Functor f => (a -> b) -> f a -> f b */ \
    template<typename Ret, typename Arg, typename... Args> \
    static constexpr F<remove_f0_t<function_t<Ret(Args...)> > > \
    fmap(function_t<Ret(Arg, Args...)> const& f, F<fdecay<Arg> > const& v);

template<typename F, typename FT>
struct fmap_result_type;

template<typename F, typename FT>
using fmap_result_type_t = typename fmap_result_type<F, FT>::type;

template<typename F, typename Ret, typename Arg, typename... Args>
struct fmap_result_type<F, function_t<Ret(Arg, Args...)> > {
    static_assert(is_functor<F>::value, "Should be a Functor");
    static_assert(is_same_as<value_type_t<F>, Arg>::value, "Should be the same");
    using type = typename F::template type<remove_f0_t<function_t<Ret(Args...)> > >;
};

template<typename F, typename FT>
using fmap_type = typename std::enable_if<
    is_functor<F>::value && std::is_same<value_type_t<F>, first_argument_type_t<function_t<FT> > >::value,
    fmap_result_type_t<F, function_t<FT> >
>::type;

#define FMAP_TYPE_(F, FT) BOOST_IDENTITY_TYPE((fmap_type<F, FT>))
#define FMAP_TYPE(F, FT) typename FMAP_TYPE_(F, FT)

// <$> fmap :: Functor f => (a -> b) -> f a -> f b
DEFINE_FUNCTION_2(2, FMAP_TYPE(T0, T1), fmap, function_t<T1> const&, f, T0 const&, v,
    return Functor_t<T0>::fmap(f, v);)

template<typename F, typename FT>
constexpr fmap_type<F, FT> operator/(function_t<FT> const& f, F const& v){
    return fmap(f, v);
}

// liftA == fmap
DEFINE_FUNCTION_2(2, FMAP_TYPE(T0, T1), liftA, function_t<T1> const&, f, T0 const&, v,
    return fmap(f, v);)

/*
-- | Replace all locations in the input with the same value.
--The default definition is @'fmap' . 'const'@, but this may be
-- overridden with a more efficient version.
(<$)        ::a->f b->f a
(<$) = fmap . const
*/
template<typename F, typename T>
constexpr functor_type<F, typeof_t<F, T> > left_fmap(T const& v, F const& f){
    return _const_<value_type_t<F> >(v) / f;
}

template<typename F, typename T>
constexpr functor_type<F, function_t<typeof_t<F, T>(F const&)> > _left_fmap(T const& v){
    return [v](F const& f){
        return left_fmap(v, f);
    };
}

template<typename F, typename T>
constexpr functor_type<F, typeof_t<F, T> > operator/=(T const& v, F const& f){
    return left_fmap(v, f);
}

_FUNCPROG_END
