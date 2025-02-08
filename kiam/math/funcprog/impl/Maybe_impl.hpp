#pragma once

#include "../Maybe.hpp"

_FUNCPROG_BEGIN

// Constructors
template<typename T>
constexpr Maybe<fdecay<T> > Just(T const& value){
    return value;
}

template<typename A>
constexpr Maybe<A> Nothing(){
    return Maybe<A>();
}

template<typename A>
constexpr f0<Maybe<A> > _Nothing(){
    return [](){ return Nothing<A>(); };
}

// Maybe
// maybe::b -> (a->b)->Maybe a->b
// maybe n _ Nothing = n
// maybe _ f(Just x) = f x
FUNCTION_TEMPLATE(2) constexpr T1 maybe(T1 const& default_value, function_t<T1(T0)> const& f, Maybe<fdecay<T0> > const& v) {
    return v ? f(v.value()) : default_value;
}

template<typename T>
bool constexpr isJust(Maybe<T> const& value){
    return (bool)value;
}

template<typename T>
bool constexpr isNothing(Maybe<T> const& value){
    return !value;
}

template<typename T>
T constexpr fromJust(Maybe<T> const& value)
{
    if (!value)
        throw maybe_nothing_error("fromJust");
    return value.value();
}

FUNCTION_TEMPLATE(1) constexpr T0 fromMaybe(T0 const& default_value, Maybe<T0> const& value) {
    return value ? value.value() : default_value;
}

template<typename A>
constexpr A fromMaybe(f0<A> const& f, Maybe<A> const& value){
    return value ? value.value() : *f;
}

template<typename A>
constexpr function_t<A(Maybe<A> const&)>
_fromMaybe(f0<A> const& f){
    return [f](Maybe<A> const& value){
        return fromMaybe(f, value);
    };
}

// Functor
template<typename Ret, typename Arg, typename... Args>
constexpr Maybe<remove_f0_t<function_t<Ret(Args...)> > >
Functor<_Maybe>::fmap(function_t<Ret(Arg, Args...)> const& f, Maybe<fdecay<Arg> > const& v)
{
    using result_type = Maybe<remove_f0_t<function_t<Ret(Args...)> > >;
    return v ? result_type(invoke_f0(f << v.value())) : result_type();
}

// Applicative
// pure = Just
template<typename A>
constexpr Maybe<fdecay<A> >
Applicative<_Maybe>::pure(A const& x){
    return Just(x);
}

template<typename Ret, typename Arg, typename... Args>
constexpr Maybe<remove_f0_t<function_t<Ret(Args...)> > >
Applicative<_Maybe>::apply(Maybe<function_t<Ret(Arg, Args...)> > const& f, Maybe<fdecay<Arg> > const& v){
    return f ? super::fmap(f.value(), v) : Nothing<remove_f0_t<function_t<Ret(Args...)> > >();
}

// Monad
template<typename Ret, typename Arg, typename... Args>
constexpr remove_f0_t<function_t<Maybe<Ret>(Args...)> >
Monad<_Maybe>::mbind(Maybe<fdecay<Arg> > const& m, function_t<Maybe<Ret>(Arg, Args...)> const& f)
{
    using result_type = Maybe<remove_f0_t<function_t<Ret(Args...)> > >;
    return m ? result_type(invoke_f0(f << m.value())) : result_type();
}

// Semigroup
template<typename A>
constexpr semigroup_type<A, Maybe<A> >
Semigroup<_Maybe>::sg_op(Maybe<A> const& x, Maybe<A> const& y){
    return !x ? y : !y ? x : Just(x.value() % y.value());
}

//stimesMaybe :: (Integral b, Semigroup a) => b -> Maybe a -> Maybe a
//stimesMaybe _ Nothing = Nothing
//stimesMaybe n (Just a) = case compare n 0 of
//    LT -> errorWithoutStackTrace "stimes: Maybe, negative multiplier"
//    EQ -> Nothing
//    GT -> Just (stimes n a)template<typename A>
template<typename A>
constexpr semigroup_type<A, Maybe<A> >
Semigroup<_Maybe>::stimes(int n, Maybe<A> const& m)
{
    assert(n >= 0);
    if (n < 0) throw maybe_error("stimes: Maybe, negative multiplier");
    return !m || n == 0 ? Nothing<A>() : Just(Semigroup_t<A>::stimes(n, m.value()));
}

// Foldable
// foldl :: (b -> a -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
Foldable<_Maybe>::foldl(function_t<Ret(B, A)> const& f, Ret const& z, Maybe<fdecay<A> > const& x){
    return x ? f(z, x.value()) : z;
}

// foldl1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
Foldable<_Maybe>::foldl1(function_t<A(Arg1, Arg2)> const& f, Maybe<A> const& x)
{
    if (!x)
        throw maybe_nothing_error("foldl1");
    return x.value();
}

// foldr :: (a -> b -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
Foldable<_Maybe>::foldr(function_t<Ret(A, B)> const& f, Ret const& z, Maybe<fdecay<A> > const& x){
    return x ? f(x.value(), z) : z;
}

// foldr1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
Foldable<_Maybe>::foldr1(function_t<A(Arg1, Arg2)> const& f, Maybe<A> const& x)
{
    if (!x)
        throw maybe_nothing_error("foldr1");
    return x.value();
}

// Traversable
// traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
// instance Traversable Maybe where
//    traverse :: Applicative f => (a -> f b) -> Maybe a -> f (Maybe b)
//    traverse _ Nothing = pure Nothing
//    traverse f (Just x) = Just <$> f x
template<typename AP, typename Arg>
constexpr applicative_type<AP, typeof_t<AP, Maybe<value_type_t<AP> > > >
Traversable<_Maybe>::traverse(function_t<AP(Arg)> const& f, Maybe<fdecay<Arg> > const& x){
    //return x ? _(Just<value_type_t<AP> >) / f(x.value()) : Applicative_t<AP>::pure(Nothing<value_type_t<AP> >());
    return x ? Functor_t<AP>::fmap(_(Just<value_type_t<AP> >), f(x.value())) :
        Applicative_t<AP>::pure(Nothing<value_type_t<AP> >());
}

// MonadError
// catchError :: m a -> (e -> m a) -> m a
// catchError Nothing f = f ()
// catchError x       _ = x
template<typename A>
constexpr Maybe<A>
MonadError<_Maybe>::catchError(Maybe<A> const& x, function_t<Maybe<A>(error_type<A> const&)> const& f){
    return x ? x : f(error_type<A>());
}

// MonadZip
// mzipWith :: (a -> b -> c) -> m a -> m b -> m c
// mzipWith = liftM2
template<typename A, typename B, typename C, typename ArgA, typename ArgB>
constexpr Maybe<C>
MonadZip<_Maybe>::mzipWith(function_t<C(ArgA, ArgB)> const& f, Maybe<A> const& ma, Maybe<B> const& mb)
{
    static_assert(is_same_as_v<ArgA, A>, "Should be the same");
    static_assert(is_same_as_v<ArgB, B>, "Should be the same");
    return Monad<_Maybe>::liftM2(f)(ma, mb);
}

_FUNCPROG_END

namespace std {

template<typename T>
ostream& operator<<(ostream& os, _FUNCPROG::Maybe<T> const& mv){
    return mv ? os << "Just(" << mv.value() << ')' : os << "Nothing";
}

template<typename T>
wostream& operator<<(wostream& os, _FUNCPROG::Maybe<T> const& mv){
    return mv ? os << L"Just(" << mv.value() << L')' : os << L"Nothing";
}

ostream& operator<<(ostream& os, _FUNCPROG::Maybe<string> const& mv){
    return mv ? os << "Just(\"" << mv.value() << "\")" : os << "Nothing";
}

wostream& operator<<(wostream& os, _FUNCPROG::Maybe<wstring> const& mv){
    return mv ? os << L"Just(\"" << mv.value() << L"\")" : os << L"Nothing";
}

ostream& operator<<(ostream& os, _FUNCPROG::Maybe<_FUNCPROG::f0<string> > const& mv){
    return mv ? os << "Just(\"" << mv.value()() << "\")" : os << "Nothing";
}

wostream& operator<<(wostream& os, _FUNCPROG::Maybe<_FUNCPROG::f0<wstring> > const& mv){
    return mv ? os << L"Just(\"" << mv.value()() << L"\")" : os << L"Nothing";
}

template<typename A>
std::ostream& operator<<(std::ostream& os, _FUNCPROG::EmptyData<A> const&){
    return os << "()";
}

template<typename A>
std::wostream& operator<<(std::wostream& os, _FUNCPROG::EmptyData<A> const&){
    return os << L"()";
}

} // namespace std
