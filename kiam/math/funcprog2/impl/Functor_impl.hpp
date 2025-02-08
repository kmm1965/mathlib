#pragma once

#include "../Functor.hpp"

_FUNCPROG2_BEGIN

/*
-- | Replace all locations in the input with the same value.
--The default definition is @'fmap' . 'const'@, but this may be
-- overridden with a more efficient version.
(<$) :: a -> f b -> f a
(<$) = fmap . const
*/
template<typename _F>
template<typename F, typename A>
__DEVICE constexpr typeof_t<_F, A> _Functor<_F>::left_fmap(A const& v, F const& f){
    static_assert(_is_same_functor_v<_F, base_class_t<F> >, "Should be the same functor");
    return _const_<value_type_t<F> >(v) / f;
}

template<typename F, typename FuncImpl, typename Ret, typename Arg0, typename... Args>
__DEVICE constexpr fmap_type<F, Ret, Arg0, Args...> fmap(function2<Ret(Arg0 const&, Args...), FuncImpl> const& f, F const& v) {
    return Functor_t<F>::fmap(f, v);
}

// liftA == fmap
FUNCTION_TEMPLATE_ARGS(4) constexpr FMAP_TYPE(T0, T2, T3) liftA(FUNCTION2(T2(T3, Args...), T1) const& f, T0 const& v) {
    return fmap(f, v);
}

FUNCTION_TEMPLATE(2) constexpr FUNCTOR_TYPE(T0, TYPEOF_T(T0, T1)) left_fmap(T1 const& v, T0 const& f) {
    return Functor_t<T0>::left_fmap(v, f);
}

template<typename F, typename T>
constexpr functor_type<F, typeof_t<F, T> > operator/=(T const& v, F const& f){
    return left_fmap(v, f);
}

_FUNCPROG2_END
