#pragma once

#include "Maybe_fwd.hpp"
#include "MonadPlus.hpp"
#include "Monoid.hpp"
#include "Foldable.hpp"
#include "Traversable.hpp"
#include "MonadError.hpp"
#include "Monad/MonadFix.hpp"
#include "Monad/MonadZip.hpp"

_FUNCPROG_BEGIN

class maybe_error : monad_error
{
public:
    maybe_error(const char* msg) : monad_error(msg) {}
};

class maybe_nothing_error : maybe_error
{
public:
    maybe_nothing_error(const char* msg) : maybe_error(msg) {}
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

    Maybe() {}
    Maybe(value_type const& value) : super(value) {}
    Maybe(f0<value_type> const& fvalue) : super(fvalue) {}
    Maybe(fdata<value_type> const& value) : super(value) {}
    Maybe(Maybe const& mvalue) : super(mvalue ? super(mvalue.value()) : super()) {}
    Maybe(f0<Maybe> const& fmvalue) : super(*fmvalue ? super((*fmvalue).value()) : super()) {}

    value_type value() const
    {
        assert(super::operator bool());
        return super::value();
    }
};

// Functor
IMPLEMENT_FUNCTOR(_Maybe);

template<>
struct Functor<_Maybe>
{
    DECLARE_FUNCTOR_CLASS(Maybe)
};

// Applicative
IMPLEMENT_APPLICATIVE(_Maybe);

template<>
struct Applicative<_Maybe> : Functor<_Maybe>
{
    typedef Functor<_Maybe> super;

    DECLARE_APPLICATIVE_CLASS(Maybe)
};

// Monad
IMPLEMENT_MONAD(_Maybe);

template<>
struct Monad<_Maybe> : Applicative<_Maybe>
{
    typedef Applicative<_Maybe> super;

    DECLARE_MONAD_CLASS(Maybe, _Maybe)

    static Maybe<const char*> fail(const char*) {
        return Nothing<const char*>();
    }
};

// Alternative
IMPLEMENT_ALTERNATIVE(_Maybe);

template<>
struct Alternative<_Maybe>
{
    DECLARE_ALTERNATIVE_CLASS(Maybe)
};

// MonadPlus
IMPLEMENT_MONADPLUS(_Maybe);

template<>
struct MonadPlus<_Maybe> : Monad<_Maybe>, Alternative<_Maybe>
{
    using super = Alternative<_Maybe>;

    DECLARE_MONADPLUS_CLASS(Maybe)
};

// Semigroup
IMPLEMENT_SEMIGROUP_COND(Maybe);

template<>
struct Semigroup<_Maybe>
{
    template<typename A>
    static semigroup_type<A, Maybe<A> > semigroup_op(Maybe<A> const& x, Maybe<A> const& y);

    template<typename A>
    static semigroup_type<A, Maybe<A> > stimes(int n, Maybe<A> const&);
};

// Monoid
template<typename A> struct is_monoid<Maybe<A> > : is_semigroup<A> {};

template<>
struct Monoid<_Maybe> : _Monoid, Semigroup<_Maybe>
{
    template<typename A>
    static semigroup_type<A, Maybe<A> > mempty() {
        return Nothing<A>();
    }
};

// Foldable
IMPLEMENT_FOLDABLE(_Maybe);

template<>
struct Foldable<_Maybe> : Monoid<_Maybe>
{
    DECLARE_FOLDABLE_CLASS(Maybe)
};

// Traversable
IMPLEMENT_TRAVERSABLE(_Maybe);

template<>
struct Traversable<_Maybe>
{
    DECLARE_TRAVERSABLE_CLASS(Maybe)
};

// MonadError
template<>
struct MonadError<_Maybe> : MonadError_base<_Maybe>
{
    template<typename A>
    using error_type = EmptyData<A>;

    DECLARE_MONADERROR_CLASS(Maybe)
};

// MonadFix
/*
instance MonadFix Maybe where
    mfix f = let a = f (unJust a) in a
             where unJust (Just x) = x
                   unJust Nothing  = errorWithoutStackTrace "mfix Maybe: Nothing"
*/
template<>
struct MonadFix<_Maybe>
{
    // mfix :: (a -> m a) -> m a
    template<typename Arg>
    static Maybe<fdecay<Arg> > mfix(function_t<Maybe<fdecay<Arg> >(Arg)> const& f)
    {
        using A = fdecay<Arg>;
        const function_t<A(Maybe<A> const&)> unJust = [](Maybe<A> const& m)
        {
            if (!m) throw maybe_error("mfix Maybe: Nothing");
            return m.value();
        };
        return Nothing<A>();
    }
};

// MonadZip
template<>
struct MonadZip<_Maybe> : _MonadZip<MonadZip<_Maybe> >
{
    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith = liftM2
    template<typename A, typename B, typename C, typename ArgA, typename ArgB>
    static Maybe<C> mzipWith(function_t<C(ArgA, ArgB)> const& f, Maybe<A> const& ma, Maybe<B> const& mb)
    {
        static_assert(is_same_as<ArgA, A>::value, "Should be the same");
        static_assert(is_same_as<ArgB, B>::value, "Should be the same");
        return liftM2(f, ma, mb);
    }
};

template<typename T>
struct is_maybe : std::false_type {};

template<typename A>
struct is_maybe<Maybe<A> > : std::true_type {};

_FUNCPROG_END

#include "detail/Maybe_impl.hpp"
