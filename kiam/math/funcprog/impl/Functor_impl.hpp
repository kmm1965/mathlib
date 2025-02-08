#pragma once

#include "../Functor.hpp"

_FUNCPROG_BEGIN

/*
-- | Replace all locations in the input with the same value.
--The default definition is @'fmap' . 'const'@, but this may be
-- overridden with a more efficient version.
(<$) :: a -> f b -> f a
(<$) = fmap . const
*/
template<typename _F>
template<typename F, typename A>
constexpr typeof_t<_F, A> _Functor<_F>::left_fmap(A const& v, F const& f){
    static_assert(_is_same_functor_v<_F, base_class_t<F> >, "Should be the same functor");
    return _const_<value_type_t<F> >(v) / f;
}

// <$> fmap :: Functor f => (a -> b) -> f a -> f b
FUNCTION_TEMPLATE_ARGS(3) constexpr FMAP_TYPE(T0, T1, T2) fmap(function_t<T1(T2, Args...)> const& f, T0 const& v) {
    return Functor_t<T0>::fmap(f, v);
}

// liftA == fmap
FUNCTION_TEMPLATE_ARGS(3) constexpr FMAP_TYPE(T0, T1, T2) liftA(function_t<T1(T2, Args...)> const& f, T0 const& v) {
    return fmap(f, v);
}

FUNCTION_TEMPLATE(2) constexpr FUNCTOR_TYPE(T0, TYPEOF_T(T0, T1)) left_fmap(T1 const& v, T0 const& f){
    return Functor_t<T0>::left_fmap(v, f);
}

_FUNCPROG_END
