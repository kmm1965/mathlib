#pragma once

#ifdef __CUDACC__
#error "CUDA compoiler is not supported for funcprog functionality"
#endif

#include "funcprog/Monad.hpp"

_KIAM_MATH_BEGIN

// Functor
template<class EO, typename Ret, typename Arg, typename... Args>
struct fmap_evaluable_object : evaluable_object<typename EO::tag_type, fmap_evaluable_object<EO, Ret, Arg, Args...> >
{
    static_assert(funcprog::is_same_as<funcprog::value_type_t<EO>, Arg>::value, "Should be the same");

    using value_type = funcprog::remove_f0_t<funcprog::function_t<Ret(Args...)> >;
    using function_type = funcprog::function_t<Ret(Arg, Args...)>;
    using eobj_type = EOBJ(EO);

    fmap_evaluable_object(function_type const& f, eobj_type const& eobj) : f(f), eobj_proxy(eobj.get_proxy()){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return funcprog::invoke_f0(funcprog::operator<<(f, eobj_proxy[i]));
    }

private:
    const function_type f;
    const typename EO::proxy_type eobj_proxy;
};

template<class EO, typename Ret, typename Arg, typename... Args>
fmap_evaluable_object<EO, Ret, Arg, Args...>
operator/(funcprog::function_t<Ret(Arg, Args...)> const& f, EOBJ(EO) const& eobj){
    return fmap_evaluable_object<EO, Ret, Arg, Args...>(f, eobj);
}

// Applicative
template<typename TAG, typename T>
struct pure_evaluable_object : evaluable_object<TAG, pure_evaluable_object<TAG, T> >
{
    using value_type = T;

    pure_evaluable_object(value_type const& value) : value(value){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t) const {
        return value;
    }

private:
    const value_type value;
};

template<class EO_F, class EO>
struct apply_evaluable_object : evaluable_object<typename EO::tag_type, apply_evaluable_object<EO_F, EO> >
{
    static_assert(std::is_same<typename EO::tag_type, typename EO_F::tag_type>::value, "The tag should be the same");
    static_assert(funcprog::is_function< get_value_type_t<EO_F> >::value, "Should be a function");
    static_assert(funcprog::is_same_as<get_value_type_t<EO>, funcprog::first_argument_type_t<get_value_type_t<EO_F> > >::value, "Should be the same");

    using value_type = funcprog::remove_f0_t<funcprog::remove_first_arg_t<get_value_type_t<EO_F> > >;
#ifdef __CUDACC__
    static_assert(!funcprog::is_function<value_type>::value, "Should not be a function");
#endif
    using eobj_f_type = evaluable_object<typename EO_F::tag_type, EO_F, typename EO_F::proxy_type>;
    using eobj_type = EOBJ(EO);

    apply_evaluable_object(eobj_f_type const& eobj_f, eobj_type const& eobj) : eobj_f_proxy(eobj_f.get_proxy()), eobj_proxy(eobj.get_proxy()){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const
    {
#ifdef __CUDACC__
        return eobj_f_proxy[i](eobj_proxy[i]);
#else
        using funcprog::operator<<;
        return funcprog::invoke_f0(eobj_f_proxy[i] << eobj_proxy[i]);
#endif
    }

private:
    const typename EO_F::proxy_type eobj_f_proxy;
    const typename EO::proxy_type eobj_proxy;
};

template<class EO_F, class EO>
typename std::enable_if<
    funcprog::is_function<get_value_type_t<EO_F> >::value,
    apply_evaluable_object<EO_F, EO>
>::type operator*(evaluable_object<typename EO_F::tag_type, EO_F, typename EO_F::proxy_type> const& eobj_f, EOBJ(EO) const& eobj){
    return apply_evaluable_object<EO_F, EO>(eobj_f, eobj);
}

template<class EO_F, class EO>
typename std::enable_if<
    funcprog::is_function<get_value_type_t<EO_F> >::value,
    apply_evaluable_object<EO_F, EO>
>::type operator_qq(evaluable_object<typename EO_F::tag_type, EO_F, typename EO_F::proxy_type> const& eobj_f, EOBJ(EO) const& eobj){
    return apply_evaluable_object<EO_F, EO>(eobj_f, eobj);
}

// Monad
template<class EO, class EORet, typename Arg>
struct mbind_evaluable_object : evaluable_object<typename EO::tag_type, mbind_evaluable_object<EO, EORet, Arg> >
{
    static_assert(std::is_same<typename EO::tag_type, typename EORet::tag_type>::value, "Tags should be the same");
    static_assert(funcprog::is_same_as<get_value_type_t<EO>, Arg>::value, "Should be the same");

    using value_type = funcprog::value_type_t<EORet>;
    using function_type = funcprog::function_t<EORet(Arg)>;

    mbind_evaluable_object(EOBJ(EO) const& eobj, function_type const& f) : eobj_proxy(eobj.get_proxy()), f(f){}

    __DEVICE
    CONSTEXPR value_type operator[](size_t i) const {
        return f(eobj_proxy[i])[i];
    }

private:
    const typename EO::proxy_type eobj_proxy;
    const function_type f;
};

template<class EO, class EORet, typename Arg>
static typename std::enable_if<
    std::is_same<typename EO::tag_type, typename EORet::tag_type>::value && funcprog::is_same_as<funcprog::value_type_t<EO>, Arg>::value,
    mbind_evaluable_object<EO, EORet, Arg>
>::type operator>>=(EOBJ(EO) const& eobj, funcprog::function_t<EORet(Arg)> const& f){
    return mbind_evaluable_object<EO, EORet, Arg>(eobj, f);
}

template<class EO, class EORet, typename Arg>
static typename std::enable_if<
    std::is_same<typename EO::tag_type, typename EORet::tag_type>::value && funcprog::is_same_as<funcprog::value_type_t<EO>, Arg>::value,
    mbind_evaluable_object<EO, EORet, Arg>
>::type operator<<=(funcprog::function_t<EORet(Arg)> const& f, EOBJ(EO) const& eobj){
    return eobj >>= f;
}

namespace funcprog {

    // Functor
    template<typename TAG>
    struct _is_functor<_KIAM_MATH::_evaluable_object<TAG> > : std::true_type {};

    template<typename TAG>
    struct Functor<_KIAM_MATH::_evaluable_object<TAG> >
    {
        // <$> fmap :: Functor f => (a -> b) -> f a -> f b
        template<class EO, typename Ret, typename Arg, typename... Args>
        static _KIAM_MATH::fmap_evaluable_object<EO, Ret, Arg, Args...>
        fmap(function_t<Ret(Arg, Args...)> const& f, EOBJ(EO) const& eobj)
        {
            static_assert(std::is_same<TAG, typename EO::tag_type>::value, "Tags should be the same");
            return _KIAM_MATH::fmap_evaluable_object<EO, Ret, Arg, Args...>(f, eobj);
        }
    };

    // Applicative
    template<typename TAG>
    struct _is_applicative<_KIAM_MATH::_evaluable_object<TAG> > : std::true_type {};

    template<typename TAG>
    struct Applicative<_KIAM_MATH::_evaluable_object<TAG> > : Functor<_KIAM_MATH::_evaluable_object<TAG> >
    {
        using super = Functor<_KIAM_MATH::_evaluable_object<TAG> >;

        template<typename T>
        static _KIAM_MATH::pure_evaluable_object<TAG, T> pure(T const& value){
            return _KIAM_MATH::pure_evaluable_object<TAG, T>(value);
        }

        // <*> :: Applicative f => f (a -> b) -> f a -> f b
        template<class EO, class EO_F>
        static _KIAM_MATH::apply_evaluable_object<EO_F, EO>
        apply(_KIAM_MATH::evaluable_object<typename EO_F::tag_type, EO_F, typename EO_F::proxy_type> const& eobj_f, EOBJ(EO) const& eobj)
        {
            static_assert(std::is_same<TAG, typename EO::tag_type>::value, "Tags should be the same");
            return _KIAM_MATH::apply_evaluable_object<EO_F, EO>(eobj_f, eobj);
        }
    };

    // Monad
    template<typename TAG>
    struct _is_monad<_KIAM_MATH::_evaluable_object<TAG> > : std::true_type {};

    template<typename TAG>
    struct Monad<_KIAM_MATH::_evaluable_object<TAG> > : Applicative<_KIAM_MATH::_evaluable_object<TAG> >
    {
        using super = Applicative<_KIAM_MATH::_evaluable_object<TAG> >;

        //template<typename T>
        //using liftM_type = _KIAM_MATH::mbind_evaluable_object<EO, _KIAM_MATH::pure_evaluable_object<TAG, T>, T>

        template<typename T>
        static _KIAM_MATH::pure_evaluable_object<TAG, T> mreturn(T const& value){
            return super::pure(value);
        }

        template<class EO, class EORet, typename Arg>
        static typename std::enable_if<
            std::is_same<TAG, typename EO::tag_type>::value &&
            std::is_same<TAG, typename EORet::tag_type>::value &&
            is_same_as<value_type_t<EO>, Arg>::value,
            _KIAM_MATH::mbind_evaluable_object<EO, EORet, Arg>
        >::type mbind(EOBJ(EO) const& eobj, function_t<EORet(Arg)> const& f){
            return _KIAM_MATH::mbind_evaluable_object<EO, EORet, Arg>(eobj, f);
        }

        template<typename Ret, typename Arg, typename... Args>
        using lift_type = function_t<_KIAM_MATH::pure_evaluable_object<TAG, remove_f0_t<function_t<Ret(Args...)> > >(Arg)>;

        template<typename Ret, typename Arg, typename... Args>
        static lift_type<Ret, Arg, Args...> lift(function_t<Ret(Arg, Args...)> const& f){
            return [f](Arg arg){
                return _KIAM_MATH::pure_evaluable_object<TAG, remove_f0_t<function_t<Ret(Args...)> > >(invoke_f0(f << arg));
            };
        }
    };

    template<typename EO, typename Ret, typename Arg>
    _KIAM_MATH::mbind_evaluable_object<EO, _KIAM_MATH::pure_evaluable_object<typename EO::tag_type, Ret>, fdecay<Arg> const&>
        liftM(function_t<Ret(Arg)> const& f, EOBJ(EO) const& eobj)
    {
        return eobj >>= _([f](value_type_t<EO> const& x){
            return Monad<_KIAM_MATH::_evaluable_object<typename EO::tag_type> >::mreturn(f(x));
        });
    }

} // namespace funcprog

_KIAM_MATH_END
