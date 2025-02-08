#pragma once

#include "../Partial.hpp"

_FUNCPROG_BEGIN

template<typename R, typename A>
constexpr Partial<R, A> Partial_(function_t<Maybe<A>(R const&)> const& f){
    return f;
}

template<typename R, typename A>
constexpr function_t<Maybe<A>(R const&)> getPartial(Partial<R, A> const& partial){
    return partial.get();
}

// Functor
//instance Functor (Partial r) where
//    fmap f (Partial g) = Partial (fmap f . g)
template<typename R>
template<typename Ret, typename Arg, typename... Args>
constexpr Partial<R, remove_f0_t<function_t<Ret(Args...)> > >
Functor<_Partial<R> >::fmap(function_t<Ret(Arg, Args...)> const& f, Partial<R, fdecay<Arg> > const& p){
    return Functor<_Maybe>::liftA(f) & p.get();
}

// Applicative
// pure x = Partial (\_ -> Just x)
template<typename R>
template<typename A>
constexpr Partial<R, A>
Applicative<_Partial<R> >::pure(A const& x){
    return [x](R const&){ return Just(x); };
}

// Partial f <*> Partial g = Partial $ \x -> f x <*> g x
template<typename R>
template<typename Ret, typename Arg, typename... Args>
constexpr Partial<R, remove_f0_t<function_t<Ret(Args...)> > >
Applicative<_Partial<R> >::apply(Partial<R, function_t<Ret(Arg, Args...)> > const& x, Partial<R, fdecay<Arg> > const& y){
    return [x, y](R const& r){ return MonadPlus<_Maybe>::apply(x.get()(r), y.get()(r)); };
}

// Monad
// Partial f >>= k = Partial $ \r -> do { x <- f r; getPartial (k x) r }
template<typename R>
template<typename Ret, typename Arg, typename... Args>
constexpr remove_f0_t<function_t<Partial<R, Ret>(Args...)> >
Monad<_Partial<R> >::mbind(Partial<R, fdecay<Arg> > const& m, function_t<Partial<R, Ret>(Arg, Args...)> const& k){
    return [m, k](R const& r){
        return _do(x, m.get()(r), return k(x).get()(r);); };
}

// MonadPlus
// mzero = Partial (const Nothing)
template<typename R>
template<typename A>
constexpr Partial<R, A>
MonadPlus<_Partial<R> >::mzero(){
    return _const_<R>(Nothing<A>());
}

// Partial f `mplus` Partial g = Partial $ \x -> f x `mplus` g x
template<typename R>
template<typename A>
constexpr Partial<R, A>
MonadPlus<_Partial<R> >::mplus(Partial<R, A> const& x, Partial<R, A> const& y){
    return [x, y](R const& r){ return MonadPlus<_Maybe>::mplus(x.get()(r), y.get()(r)); };
}

// Alternative
// empty = Partial (const Nothing)
template<typename R>
template<typename A>
constexpr Partial<R, A>
Alternative<_Partial<R> >::empty(){
    return _const_<R>(Nothing<A>());
}

// Partial f <|> Partial g = Partial $ \x -> f x <|> g x
template<typename R>
template<typename A>
constexpr Partial<R, A>
Alternative<_Partial<R> >::alt_op(Partial<R, A> const& x, Partial<R, A> const& y){
    return [x, y](R const& r){ return Alternative<_Maybe>::alt_op(x.get()(r), y.get()(r)); };
}

// Monoid
// mempty = mzero
template<typename R>
template<typename A>
constexpr Partial<R, A>
Monoid<_Partial<R> >::mempty(){
    return MonadPlus<_Partial<R> >::template mzero<A>();
}

// mappend = mplus
template<typename R>
template<typename A>
constexpr Partial<R, A>
Monoid<_Partial<R> >::mappend(Partial<R, A> const& x, Partial<R, A> const& y){
    return MonadPlus<_Partial<R> >::mplus(x, y);
}

_FUNCPROG_END
