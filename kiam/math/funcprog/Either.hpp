#pragma once

#include "fwd/Either_fwd.hpp"
#include "Monad.hpp"
#include "MonadError.hpp"

_FUNCPROG_BEGIN

template<typename A>
struct _is_functor<_Either<A> > : std::true_type {};

template<typename A, typename B>
struct is_functor<Either<A, B> > : std::true_type {};

template<typename A>
struct _is_applicative<_Either<A> > : std::true_type {};

template<typename A, typename B>
struct is_applicative<Either<A, B> > : std::true_type {};

template<typename A>
struct _is_monad<_Either<A> > : std::true_type {};

template<typename A, typename B>
struct is_monad<Either<A, B> > : std::true_type {};

template<typename A>
struct _Either
{
    using base_class = _Either;

    template<typename B>
    using type = Either<A, B>;
};

template<>
struct _Either<void>
{
    using base_class = _Either;

    template<typename B>
    using type = Either<void, B>;
};

template<typename A>
struct EitherBase
{
    using value_type = A;

protected:
    EitherBase(value_type const& value) : value(value){}
    EitherBase(f0<value_type> const& fvalue) : value(fvalue){}

public:
    value_type operator*() const {
        return value();
    }

private:
    const fdata<value_type> value;
};

template<typename A>
struct _Left : EitherBase<A>
{
    _Left(A const& value) : EitherBase<A>(value){}
    _Left(f0<A> const& fvalue) : EitherBase<A>(fvalue){}
};

template<typename A>
struct _Right : EitherBase<A>
{
    _Right(A const& value) : EitherBase<A>(value){}
    _Right(f0<A> const& fvalue) : EitherBase<A>(fvalue){}
};

template<typename A, typename B>
struct Either : variant_t<_Left<A>, _Right<B> >, _Either<A>
{
    using super = variant_t<_Left<A>, _Right<B> >;
    using left_type = A;
    using right_type = B;
    using value_type = B;

    Either(A const& value) : super(_Left<A>(value)){}
    Either(B const& value) : super(_Right<B>(value)){}
    Either(_Left<A> const& value) : super(value){}
    Either(_Right<B> const& value) : super(value){}
    Either(Either<A, void> const& e) : super(e.left()){}
    Either(Either<void, B> const& e) : super(e.right()){}

    const _Left<A>& left() const
    {
        assert(super::index() == Left_);
        return super::template get<_Left<A> >();
    }

    const _Right<B>& right() const
    {
        assert(super::index() == Right_);
        return super::template get<_Right<B> >();
    }
};

template<typename A>
using SEither = Either<A, String>;

// Functor
template<typename A>
struct Functor<_Either<A> > : _Functor<_Either<A> >
{
    // <$> fmap :: Functor f => (a -> b) -> f a -> f b
    // fmap _ (Left x) = Left x
    // fmap f (Right y) = Right (f y)
    template<typename Ret, typename Arg, typename... Args>
    static constexpr Either<A, remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, Either<A, fdecay<Arg> > const& x);
};

// Applicative
template<typename A>
struct Applicative<_Either<A> > : Functor<_Either<A> >, _Applicative<_Either<A> >
{
    using super = Functor<_Either<A> >;

    // pure = Right
    template<typename T>
    static constexpr Either<A, fdecay<T> > pure(T const& x);

    // <*> :: Applicative f => f (a -> b) -> f a -> f b
    template<typename Ret, typename Arg, typename... Args>
    static constexpr Either<A, remove_f0_t<function_t<Ret(Args...)> > >
    apply(Either<A, function_t<Ret(Arg, Args...)> > const& f, Either<A, fdecay<Arg> > const& x);
};

// Monad
template<typename A>
struct Monad<_Either<A> > : Applicative<_Either<A> >, _Monad<_Either<A> >
{
    using super = Applicative<_Either<A> >;

    template<typename C, typename Arg, typename... Args>
    static constexpr remove_f0_t<function_t<Either<A, C>(Args...)> >
    mbind(Either<A, fdecay<Arg> > const& m, function_t<Either<A, C>(Arg, Args...)> const& f);
};

// MonadError
template<typename A>
struct MonadError<_Either<A> > : _MonadError<_Either<A> >
{
    template<typename B>
    using error_type = A;

    // throwError :: e -> m a
    template<typename B>
    static constexpr Either<A, B> throwError(A const& x);

    // catchError :: m a -> (e -> m a) -> m a
    template<typename B>
    static constexpr Either<A, B> catchError(Either<A, B> const& x, function_t<Either<A, B>(A const&)> const& f);
};

template<typename A>
struct Either<A, void> : _Either<A>
{
    using left_type = A;
    using right_type = void;
    using value_type = void;

    Either(const _Left<A> &value) : value(*value){}

    size_t index() const {
        return Left_;
    }

    const _Left<A>& left() const {
        return value;
    }

private:
    const _Left<A> value;
};

template<typename B>
struct Either<void, B> : _Either<void>
{
    using left_type = void;
    using right_type = B;
    using value_type = B;

    Either(const _Right<B> &value) : value(value){}

    size_t index() const {
        return Right_;
    }

    const _Right<B>& right() const {
        return value;
    }

private:
    const _Right<B> value;
};

// Functor
template<>
struct Functor<_Either<void> > : _Functor<_Either<void> >
{
    template<typename Ret, typename Arg, typename... Args>
    static constexpr Either<void, remove_f0_t<function_t<Ret(Args...)> > >
    fmap(function_t<Ret(Arg, Args...)> const& f, Either<void, fdecay<Arg> > const& x);
};

// Applicative
template<>
struct Applicative<_Either<void> > : Functor<_Either<void> >, _Applicative<_Either<void> >
{
    using super = Functor<_Either<void> >;

    template<typename T>
    static constexpr Either<void, fdecay<T> > pure(T const& x);

    template<typename Ret, typename Arg, typename... Args>
    static constexpr Either<void, remove_f0_t<function_t<Ret(Args...)> > >
    apply(Either<void, function_t<Ret(Arg, Args...)> > const& f, Either<void, fdecay<Arg> > const& x);
};

// Monad
template<>
struct Monad<_Either<void> > : Applicative<_Either<void> >, _Monad<_Either<void> >
{
    using super = Applicative<_Either<void> >;

    template<typename C, typename Arg, typename... Args>
    static constexpr remove_f0_t<function_t<Either<void, C>(Args...)> >
    mbind(Either<void, fdecay<Arg> > const& m, function_t<Either<void, C>(Arg, Args...)> const& f);
};

_FUNCPROG_END
