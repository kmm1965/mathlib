#pragma once

#include "fwd/Maybe_fwd.hpp"
#include "Functor.hpp"
#include "Applicative.hpp"
#include "MonadFail.hpp"
#include "Monad.hpp"
#include "Alternative.hpp"
#include "MonadPlus.hpp"
#include "Semigroup.hpp"
#include "Monoid.hpp"
#include "Foldable.hpp"
#include "Traversable.hpp"
#include "MonadError.hpp"
#include "Monad/MonadZip.hpp"

_FUNCPROG_BEGIN

class maybe_error : monad_error
{
public:
    maybe_error(const char* msg) : monad_error(msg){}
};

class maybe_nothing_error : maybe_error
{
public:
    maybe_nothing_error(const char* msg) : maybe_error(msg){}
};

struct _Maybe
{
    using base_class = _Maybe;

    template<typename A>
    using type = Maybe<A>;
};

template<typename A>
struct Maybe : optional_t<fdata<A> >, _Maybe
{
    using value_type = A;
    using super = optional_t<fdata<value_type> >;

    Maybe(){}
    Maybe(value_type const& value) : super(value){}
    Maybe(f0<value_type> const& fvalue) : super(fvalue){}
    Maybe(fdata<value_type> const& value) : super(value){}
    Maybe(Maybe const& mvalue) : super(mvalue ? super(mvalue.value()) : super()){}
    Maybe(f0<Maybe> const& fmvalue) : super(*fmvalue ? super((*fmvalue).value()) : super()){}

    value_type value() const
    {
        assert(super::operator bool());
        return super::value();
    }

    // Monoid
    static Maybe mempty(){ return Maybe(); }
};

// Functor
template<>
struct Functor<_Maybe> : _Functor<_Maybe>
{
    template<typename Ret, typename Arg, typename... Args>
    static constexpr Maybe<remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, Maybe<fdecay<Arg> > const& v);
};

// Applicative
template<>
struct Applicative<_Maybe> : Functor<_Maybe>, _Applicative<_Maybe>
{
    typedef Functor<_Maybe> super;

    template<typename A>
    static constexpr Maybe<fdecay<A> > pure(A const& x);

    template<typename Ret, typename Arg, typename... Args>
    static constexpr Maybe<remove_f0_t<function_t<Ret(Args...)> > >
    apply(Maybe<function_t<Ret(Arg, Args...)> > const& f, Maybe<fdecay<Arg> > const& v);
};

// MonadFail
template<>
struct MonadFail<_Maybe>
{
    template<typename A = None>
    static constexpr Maybe<A> fail(const char*){
        return Nothing<A>();
    }
};

// Monad
template<>
struct Monad<_Maybe> : Applicative<_Maybe>, _Monad<_Maybe>
{
    //typedef Applicative<_Maybe> super;
    
    template<typename A>
    using liftM_type = _Maybe::template type<A>;

    template<typename Ret, typename Arg, typename... Args>
    static constexpr remove_f0_t<function_t<Maybe<Ret>(Args...)> >
    mbind(Maybe<fdecay<Arg> > const& m, function_t<Maybe<Ret>(Arg, Args...)> const& f);
};

// Alternative
template<>
struct Alternative<_Maybe> : _Alternative<_Maybe>
{
    template<typename A>
    // empty = Nothing
    static constexpr Maybe<A> empty(){
        return Nothing<A>();
    }

    // Nothing <|> r = r
    // l       <|> _ = l
    template<typename A>
    static constexpr Maybe<A> alt_op(Maybe<A> const& l, Maybe<A> const& r){
        return l ? l : r;
    }
};

// MonadPlus
template<>
struct MonadPlus<_Maybe> : Monad<_Maybe>, Alternative<_Maybe>, _MonadPlus<_Maybe>{};

// Semigroup
template<>
struct Semigroup<_Maybe> : _Semigroup<_Maybe>
{
    template<typename A>
    static constexpr semigroup_type<A, Maybe<A> > sg_op(Maybe<A> const& x, Maybe<A> const& y);

    //stimesMaybe :: (Integral b, Semigroup a) => b -> Maybe a -> Maybe a
    template<typename A>
    static constexpr semigroup_type<A, Maybe<A> > stimes(int n, Maybe<A> const& m);
};

// Monoid
template<>
struct Monoid<_Maybe> : Semigroup<_Maybe>, _Monoid<_Maybe>
{
    template<typename A>
    static constexpr semigroup_type<A, Maybe<A> > mempty(){
        return Nothing<A>();
    }
};

// Foldable
template<>
struct Foldable<_Maybe> : Monoid<_Maybe>, _Foldable<_Maybe>
{
    // foldl :: (b -> a -> b) -> b -> t a -> b
    template<typename Ret, typename A, typename B>
    static constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
    foldl(function_t<Ret(B, A)> const& f, Ret const& z, Maybe<fdecay<A> > const& x);

    // foldl1 :: (a -> a -> a) -> t a -> a
    template<typename A, typename Arg1, typename Arg2>
    static constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
    foldl1(function_t<A(Arg1, Arg2)> const& f, Maybe<A> const& x);

    // foldr :: (a -> b -> b) -> b -> t a -> b
    template<typename Ret, typename A, typename B>
    static constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
    foldr(function_t<Ret(A, B)> const& f, Ret const& z, Maybe<fdecay<A> > const& x);

    // foldr1 :: (a -> a -> a) -> t a -> a
    template<typename A, typename Arg1, typename Arg2>
    static constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
    foldr1(function_t<A(Arg1, Arg2)> const& f, Maybe<A> const& x);
};

// Traversable
template<>
struct Traversable<_Maybe> : _Traversable<_Maybe>
{
    // traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
    // traverse :: Applicative f => (a -> f b) -> Maybe a -> f (Maybe b)
    template<typename AP, typename Arg>
    static constexpr applicative_type<AP, typeof_t<AP, Maybe<value_type_t<AP> > > >
    traverse(function_t<AP(Arg)> const& f, Maybe<fdecay<Arg> > const& x);
};

// MonadError
template<>
struct MonadError<_Maybe> : _MonadError<_Maybe>
{
    template<typename A>
    using error_type = EmptyData<A>;

    // throwError :: e -> m a
    // throwError () = Nothing
    template<typename A>
    static constexpr Maybe<A> throwError(error_type<A> const&){
        return Nothing<A>();
    }

    // catchError :: m a -> (e -> m a) -> m a
    // catchError Nothing f = f ()
    template<typename A>
    static constexpr Maybe<A>
    catchError(Maybe<A> const& x, function_t<Maybe<A>(error_type<A> const&)> const& f);
};

// MonadZip
template<>
struct MonadZip<_Maybe> : _MonadZip<MonadZip<_Maybe> >
{
    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith = liftM2
    template<typename A, typename B, typename C, typename ArgA, typename ArgB>
    static constexpr Maybe<C>
    mzipWith(function_t<C(ArgA, ArgB)> const& f, Maybe<A> const& ma, Maybe<B> const& mb);
};

_FUNCPROG_END
