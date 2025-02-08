#pragma once

#include "fwd/Identity_fwd.hpp"
#include "Functor.hpp"
#include "Applicative.hpp"
#include "Monad.hpp"
#include "Semigroup.hpp"
#include "Monoid.hpp"
#include "Foldable.hpp"
#include "Traversable.hpp"
#include "Monad/MonadZip.hpp"

_FUNCPROG_BEGIN

struct _Identity
{
    using base_class = _Identity;

    template<typename A>
    using type = Identity<A>;
};

template<typename A>
struct Identity : _Identity
{
    using value_type = A;

    Identity(value_type const& value) : value(value){}
    Identity(f0<value_type> const& fvalue) : value(fvalue){}
    Identity(Identity const& ivalue) : value(ivalue.value){}
    Identity(f0<Identity> const& fivalue) : value((*fivalue).value){}

    constexpr value_type run() const {
        return value();
    }

private:
    const fdata<value_type> value;
};

// Functor
template<>
struct Functor<_Identity> : _Functor<_Identity>
{
    // <$> fmap :: Functor f => (a -> b) -> f a -> f b
    template<typename Ret, typename Arg, typename... Args>
    static constexpr auto fmap(function_t<Ret(Arg, Args...)> const& f, Identity<fdecay<Arg> > const& v){
        return Identity_(invoke_f0(f << v.run()));
    }
};

// Applicative
template<>
struct Applicative<_Identity> : Functor<_Identity>, _Applicative<_Identity>
{
    using super = Functor<_Identity>;

    template<typename A>
    static constexpr Identity<fdecay<A> > pure(A const& x){
        return x;
    }

    template<typename Ret, typename Arg, typename... Args>
    static constexpr auto apply(Identity<function_t<Ret(Arg, Args...)> > const& f, Identity<fdecay<Arg> > const& v){
        return super::fmap(f.run(), v);
    }
};

// Monad
template<>
struct Monad<_Identity> : Applicative<_Identity>, _Monad<_Identity>
{
    template<typename A>
    using liftM_type = _Identity::template type<A>;

    template<typename Ret, typename Arg, typename... Args>
    static constexpr auto mbind(Identity<fdecay<Arg> > const& m, function_t<Identity<Ret>(Arg, Args...)> const& f){
        return invoke_f0(f << m.run());
    }
};

// Semigroup
template<>
struct Semigroup<_Identity> : _Semigroup<_Identity>
{
    template<typename A>
    static constexpr semigroup_type<A, Identity<A> >
    sg_op(Identity<A> const& x, Identity<A> const& y);

    template<typename A>
    static constexpr semigroup_type<A, Identity<A> >
    stimes(int n, Identity<A> const& m);
};

// Moniod
template<>
struct Monoid<_Identity> : Semigroup<_Identity>, _Monoid<_Identity>
{
    template<typename A>
    static constexpr monoid_type<A, Identity<A> > mempty();
};

// Foldable
template<>
struct Foldable<_Identity> : _Foldable<_Identity>
{
    // foldl :: (b -> a -> b) -> b -> t a -> b
    template<typename Ret, typename A, typename B>
    static constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
    foldl(function_t<Ret(B, A)> const& f, Ret const& z, Identity<fdecay<A> > const& x);

    // foldl1 :: (a -> a -> a) -> t a -> a
    template<typename A, typename Arg1, typename Arg2>
    static constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
    foldl1(function_t<A(Arg1, Arg2)> const&, Identity<A> const& x);

    // foldr :: (a -> b -> b) -> b -> t a -> b
    template<typename Ret, typename A, typename B>
    static constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
    foldr(function_t<Ret(A, B)> const& f, Ret const& z, Identity<fdecay<A> > const& x);

    // foldr1 :: (a -> a -> a) -> t a -> a
    template<typename A, typename Arg1, typename Arg2>
    static constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
    foldr1(function_t<A(Arg1, Arg2)> const&, Identity<A> const& x);
};

// Traversable
template<>
struct Traversable<_Identity> : _Traversable<_Identity>
{
    // traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
    template<typename AP, typename Arg>
    static constexpr applicative_type<AP, typeof_t<AP, Identity<value_type_t<AP> > > >
    traverse(function_t<AP(Arg)> const& f, Identity<fdecay<Arg> > const& x);
};

// MonadZip
template<>
struct MonadZip<_Identity> : _MonadZip<MonadZip<_Identity> >
{
    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith = liftM2
    template<typename A, typename B, typename C, typename ArgA, typename ArgB>
    static constexpr Identity<C>
    mzipWith(function_t<C(ArgA, ArgB)> const& f, Identity<A> const& ma, Identity<B> const& mb);

    // munzip (Identity (a, b)) = (Identity a, Identity b)
    template<typename A, typename B>
    static constexpr pair_t<Identity<A>, Identity<B> >
    munzip(Identity<pair_t<A, B> > const& mab);
};

_FUNCPROG_END
