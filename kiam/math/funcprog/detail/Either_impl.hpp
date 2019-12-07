#pragma once

_FUNCPROG_BEGIN

// Constructors
template<typename A>
Either<A, void> Left(A const& value) {
	return _Left<A>(value);
}

template<typename A>
Either<A, void> Left(f0<A> const& value) {
    return _Left<A>(value());
}

template<typename B>
Either<void, B> Right(B const& value) {
	return _Right<B>(value);
}

template<typename B>
Either<void, B> Right(f0<B> const& value) {
    return _Right<B>(value());
}

// Functor
// <$> fmap :: Functor f => (a -> b) -> f a -> f b
// fmap _ (Left x) = Left x
// fmap f (Right y) = Right (f y)
template<typename A>
template<typename Ret, typename Arg, typename... Args>
Either<A, remove_f0_t<function_t<Ret(Args...)> > >
Functor<_Either<A> >::fmap(function_t<Ret(Arg, Args...)> const& f, Either<A, fdecay<Arg> > const& x)
{
    if (x.index() == Left_)
        return x.left();
    else return _Right<remove_f0_t<function_t<Ret(Args...)> > >(f << x.right().value);
}

template<typename Ret, typename Arg, typename... Args>
Either<void, remove_f0_t<function_t<Ret(Args...)> > >
Functor<_Either<void> >::fmap(function_t<Ret(Arg, Args...)> const& f, Either<void, fdecay<Arg> > const& x)
{
    assert(x.index() == Right_);
	return _Right<remove_f0_t<function_t<Ret(Args...)> > >(f << *(x.right()));
}

// Applicative
// pure = Right
template<typename A>
template<typename T>
Either<A, fdecay<T> > Applicative<_Either<A> >::pure(T const& x) {
    return Right(x);
}

template<typename T>
Either<void, fdecay<T> > Applicative<_Either<void> >::pure(T const& x) {
    return Right(x);
}

// <*> :: Applicative f => f (a -> b) -> f a -> f b
// Left  e <*> _ = Left e
// Right f <*> r = fmap f r
template<typename A>
template<typename Ret, typename Arg, typename... Args>
Either<A, remove_f0_t<function_t<Ret(Args...)> > >
Applicative<_Either<A> >::apply(Either<A, function_t<Ret(Arg, Args...)> > const& f, Either<A, fdecay<Arg> > const& x)
{
    if (f.index() == Left_)
        return f.left();
    else return super::fmap(f.right().value, x);
}

template<typename Ret, typename Arg, typename... Args>
Either<void, remove_f0_t<function_t<Ret(Args...)> > >
Applicative<_Either<void> >::apply(Either<void, function_t<Ret(Arg, Args...)> > const& f, Either<void, fdecay<Arg> > const& x){
    return super::fmap(f.right().value, x);
}

// Monad
template<typename A>
template<typename C, typename Arg, typename... Args>
remove_f0_t<function_t<Either<A, C>(Args...)> >
Monad<_Either<A> >::mbind(Either<A, fdecay<Arg> > const& m, function_t<Either<A, C>(Arg, Args...)> const& f)
{
    if (m.index() == Left_)
        return m.left();
	else return f << m.right().value;
}

template<typename C, typename Arg, typename... Args>
remove_f0_t<function_t<Either<void, C>(Args...)> >
Monad<_Either<void> >::mbind(Either<void, fdecay<Arg> > const& m, function_t<Either<void, C>(Arg, Args...)> const& f)
{
    assert(m.index() == Right_);
    return f << m.right().value;
}

// MonadError
// throwError :: e -> m a
// throwError             = Left
template<typename A>
template<typename B>
Either<A, B> MonadError<_Either<A> >::throwError(A const& x) {
	return _Left<A>(x);
}

// catchError :: m a -> (e -> m a) -> m a
// Left  l `catchError` h = h l
// Right r `catchError` _ = Right r
template<typename A>
template<typename B>
Either<A, B> MonadError<_Either<A> >::catchError(Either<A, B> const& x, function_t<Either<A, B>(A const&)> const& f) {
	return x.index() == Left_ ? f(*(x.left())) : x;
}

// either                  :: (a -> c) -> (b -> c) -> Either a b -> c
// either f _ (Left x)     =  f x
// either _ g (Right y)    =  g y
DEFINE_FUNCTION_3_ARGS(3, remove_f0_t<function_t<T2(Args...)> >, either, function_t<T2(T0, Args...)> const&, f, function_t<T2(T1, Args...)> const&, g,
    EITHER(fdecay<T0>, fdecay<T1>) const&, x,
	return invoke_f0(x.index() == Left_ ? f << *(x.left()) : g << *(x.right()));)

_FUNCPROG_END

namespace std {

template<typename A, typename B>
ostream& operator<<(ostream& os, _FUNCPROG::Either<A, B> const& v) {
	return v.index() == _FUNCPROG::Left_ ?
        os << "Left(" << (*v.left()) << ')' :
		os << "Right(" << *(v.right()) << ')';
}

template<typename A>
ostream& operator<<(ostream& os, _FUNCPROG::Either<A, void> const& v) {
    assert(v.index() == _FUNCPROG::Left_);
    return os << "Left(" << *(v.left()) << ')';
}

template<typename B>
ostream& operator<<(ostream& os, _FUNCPROG::Either<void, B> const& v) {
    assert(v.index() == _FUNCPROG::Right_);
    return os << "Right(" << *(v.right()) << ')';
}

template<typename A, typename B>
wostream& operator<<(wostream& os, _FUNCPROG::Either<A, B> const& v) {
	return v.index() == _FUNCPROG::Left_ ?
        os << L"Left(" << v.left().value << L')' :
		os << L"Right(" << v.right().value << L')';
}

template<typename A>
wostream& operator<<(wostream& os, _FUNCPROG::Either<A, void> const& v) {
    assert(v.index() == _FUNCPROG::Left_);
    return os << L"Left(" << v.left().value << L')';
}

template<typename B>
wostream& operator<<(wostream& os, _FUNCPROG::Either<void, B> const& v) {
    assert(v.index() == _FUNCPROG::Right_);
    return os << L"Right(" << v.right().value << L')';
}

} // namespace std
