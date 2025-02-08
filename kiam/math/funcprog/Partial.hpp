#pragma once

#include "fwd/Partial_fwd.hpp"
#include "Functor.hpp"
#include "Applicative.hpp"
#include "Monad.hpp"
#include "MonadPlus.hpp"
#include "Alternative.hpp"
#include "Monoid.hpp"

_FUNCPROG_BEGIN

struct __Partial {
    template<typename R>
    using base_type = _Partial<R>;
};

template<typename R>
struct _Partial : __Partial {
    using base_class = _Partial;

    template<typename A>
    using type = Partial<R, A>;
};

template<typename R, typename A>
struct Partial : _Partial<R> {
    using value_type = A;
    using func_type = function_t<Maybe<A>(R const&)>;

    Partial(func_type const& func) : func(func) {}

    func_type get() const { return func; }

private:
    func_type const func;
};

// Functor
template<typename R>
struct Functor<_Partial<R> > : _Functor<_Partial<R> >
{
    //instance Functor (Partial r) where
    //    fmap f (Partial g) = Partial (fmap f . g)
    template<typename Ret, typename Arg, typename... Args>
    static constexpr Partial<R, remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, Partial<R, fdecay<Arg> > const& p);
};

// Applicative
template<typename R>
struct Applicative<_Partial<R> > : _Applicative<_Partial<R> >
{
    // pure x = Partial (\_ -> Just x)
    template<typename A>
    constexpr Partial<R, A> pure(A const& x);

    // Partial f <*> Partial g = Partial $ \x -> f x <*> g x
    template<typename Ret, typename Arg, typename... Args>
    static constexpr Partial<R, remove_f0_t<function_t<Ret(Args...)> > >
    apply(Partial<R, function_t<Ret(Arg, Args...)> > const& x, Partial<R, fdecay<Arg> > const& y);
};

// Monad
template<typename R>
struct Monad<_Partial<R> > : _Monad<_Partial<R> >
{
    // Partial f >>= k = Partial $ \r -> do { x <- f r; getPartial (k x) r }
    template<typename Ret, typename Arg, typename... Args>
    static constexpr remove_f0_t<function_t<Partial<R, Ret>(Args...)> >
    mbind(Partial<R, fdecay<Arg> > const& m, function_t<Partial<R, Ret>(Arg, Args...)> const& k);
};

// MonadPlus
template<typename R>
struct MonadPlus<_Partial<R> > : _MonadPlus<_Partial<R> >
{
    // mzero = Partial (const Nothing)
    template<typename A>
    static constexpr Partial<R, A> mzero();

    // Partial f `mplus` Partial g = Partial $ \x -> f x `mplus` g x
    template<typename A>
    static constexpr Partial<R, A>
    mplus(Partial<R, A> const& x, Partial<R, A> const& y);
};

// Alternative
template<typename R>
struct Alternative<_Partial<R> > : _Alternative<_Partial<R> >
{
    // empty = Partial (const Nothing)
    template<typename A>
    static constexpr Partial<R, A> empty();

    // Partial f <|> Partial g = Partial $ \x -> f x <|> g x
    template<typename A>
    static constexpr Partial<R, A>
    alt_op(Partial<R, A> const& x, Partial<R, A> const& y);
};

// Monoid
template<typename R>
struct Monoid<_Partial<R> > : _Monoid<_Partial<R> >
{
    // mempty = mzero
    template<typename A>
    static constexpr Partial<R, A> mempty();

    // mappend = mplus
    template<typename A>
    static constexpr Partial<R, A> mappend(Partial<R, A> const& x, Partial<R, A> const& y);
};

_FUNCPROG_END
