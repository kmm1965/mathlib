#pragma once

#include "Identity_fwd.hpp"
#include "Monad.hpp"
#include "Foldable.hpp"
#include "Traversable.hpp"
#include "Monad/MonadFix.hpp"
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

    Identity(value_type const& value) : value(value) {}
    Identity(f0<value_type> const& fvalue) : value(fvalue) {}
    Identity(Identity const& ivalue) : value(ivalue.value) {}
    Identity(f0<Identity> const& fivalue) : value((*fivalue).value) {}
    //Identity(f0<Identity> const& fivalue) : value(_([fivalue]() { return (*fivalue).value(); })) {}

    value_type run() const {
        return value();
    }

private:
    const fdata<value_type> value;
};

template<typename T>
T runIdentity(Identity<T> const& x) {
    return x.run();
}

// Constructor
template<typename T>
Identity<T> Identity_(T const& value) {
    return value;
}

template<typename T>
Identity<T> Identity_f(f0<T> const& fvalue) {
    return fvalue;
}

// Functor
IMPLEMENT_FUNCTOR(Identity, _Identity)

template<>
struct Functor<_Identity>
{
    DECLARE_FUNCTOR_CLASS(Identity)
};

// Applicative
IMPLEMENT_APPLICATIVE(Identity, _Identity)

template<>
struct Applicative<_Identity> : Functor<_Identity>
{
    typedef Functor<_Identity> super;

    DECLARE_APPLICATIVE_CLASS(Identity)
};

// Monad
IMPLEMENT_MONAD(Identity, _Identity)

template<>
struct Monad<_Identity> : Applicative<_Identity>
{
    typedef Applicative<_Identity> super;

    DECLARE_MONAD_CLASS(Identity, _Identity)
};

// Foldable
IMPLEMENT_FOLDABLE(Identity)

template<>
struct Foldable<_Identity>
{
    DECLARE_FOLDABLE_CLASS(Identity)
};

// Traversable
IMPLEMENT_TRAVERSABLE(Identity)

template<>
struct Traversable<_Identity>
{
    DECLARE_TRAVERSABLE_CLASS(Identity)
};

template<>
struct MonadFix<_Identity>
{
    // mfix :: (a -> m a) -> m a
    // mfix f = Identity (fix (runIdentity . f))
    template<typename Arg>
    static Identity<fdecay<Arg> > mfix(function_t<Identity<fdecay<Arg> >(Arg)> const& f)
    {
        using A = fdecay<Arg>;
        //return fix<A>(_(runIdentity<A>) & f);
    }
};

// MonadZip
template<>
struct MonadZip<_Identity> : _MonadZip<MonadZip<_Identity> >
{
    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith = liftM2
    template<typename A, typename B, typename C, typename ArgA, typename ArgB>
    static Identity<C> mzipWith(function_t<C(ArgA, ArgB)> const& f, Identity<A> const& ma, Identity<B> const& mb)
    {
        static_assert(is_same_as<ArgA, A>::value, "Should be the same");
        static_assert(is_same_as<ArgB, B>::value, "Should be the same");
        return liftM2(f, ma, mb);
    }

    // munzip (Identity (a, b)) = (Identity a, Identity b)
    template<typename A, typename B>
    static pair_t<Identity<A>, Identity<B> > munzip(Identity<pair_t<A, B> > const& mab)
    {
        const auto [a, b] = mab.run();
        return std::make_pair(Identity_(a), Identity_(b));
    }
};

template<typename T>
struct is_identity : std::false_type {};

template<typename A>
struct is_identity<Identity<A> > : std::true_type {};

_FUNCPROG_END

#include "detail/Identity_impl.hpp"
