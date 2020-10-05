#pragma once

#include "Monad_impl.hpp"

_FUNCPROG_BEGIN

// Constructors
template<typename T>
Maybe<fdecay<T> > Just(T const& value) {
    return value;
}

template<typename A>
Maybe<A> Nothing() {
    return Maybe<A>();
}

template<typename A>
f0<Maybe<A> > _Nothing() {
    return []() { return Nothing<A>(); };
}

// Functor
template<typename Ret, typename Arg, typename... Args>
Maybe<remove_f0_t<function_t<Ret(Args...)> > >
Functor<_Maybe>::fmap(function_t<Ret(Arg, Args...)> const& f, Maybe<fdecay<Arg> > const& v)
{
    using result_type = Maybe<remove_f0_t<function_t<Ret(Args...)> > >;
    return v ? result_type(f << v.value()) : result_type();
}

// Applicative
// pure = Just
template<typename T>
Maybe<fdecay<T> > Applicative<_Maybe>::pure(T const& x) {
    return Just(x);
}

template<typename Ret, typename Arg, typename... Args>
Maybe<remove_f0_t<function_t<Ret(Args...)> > >
Applicative<_Maybe>::apply(Maybe<function_t<Ret(Arg, Args...)> > const& f, Maybe<fdecay<Arg> > const& v){
    return f ? super::fmap(f.value(), v) : Nothing<remove_f0_t<function_t<Ret(Args...)> > >();
}

// Monad
IMPLEMENT_MRETURN(Maybe, _Maybe)

template<typename Ret, typename Arg, typename... Args>
remove_f0_t<function_t<Maybe<Ret>(Args...)> >
Monad<_Maybe>::mbind(Maybe<fdecay<Arg> > const& m, function_t<Maybe<Ret>(Arg, Args...)> const& f)
{
    using result_type = Maybe<remove_f0_t<function_t<Ret(Args...)> > >;
    return m ? result_type(f << m.value()) : result_type();
}

// MonadPlus
IMPLEMENT_DEFAULT_MONADPLUS(Maybe, _Maybe)

// Alternative
// empty = Nothing
template<typename A>
Maybe<A> Alternative<_Maybe>::empty() {
    return Nothing<A>();
}

// Nothing <|> r = r
// l       <|> _ = l
template<typename A>
Maybe<A> Alternative<_Maybe>::alt_op(Maybe<A> const& l, Maybe<A> const&r) {
    return l ? l : r;
}

// Semigroup
template<typename A>
typename std::enable_if<is_semigroup<A>::value, Maybe<A> >::type
Semigroup<_Maybe>::semigroup_op(Maybe<A> const& x, Maybe<A> const& y) {
    return !x ? y : !y ? x : Just(x.value() % y.value());
}

template<typename A>
typename std::enable_if<is_semigroup<A>::value, Maybe<A> >::type
Semigroup<_Maybe>::stimes(int n, Maybe<A> const& m)
{
    assert(n >= 0);
    if (n < 0) throw maybe_error("stimes: Maybe, negative multiplier");
    return !m || n == 0 ? Nothing<A>() : Just(Semigroup_t<A>::stimes(n, m.value()));
}

// Foldable
DEFAULT_FOLDMAP_IMPL(Maybe, _Maybe)

// foldl :: (b -> a -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
typename std::enable_if<is_same_as<Ret, B>::value, Ret>::type
Foldable<_Maybe>::foldl(function_t<Ret(B, A)> const& f, Ret const& z, Maybe<fdecay<A> > const& x){
    return x ? f(z, x.value()) : z;
}

// foldl1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
typename std::enable_if<is_same_as<A, Arg1>::value&& is_same_as<A, Arg2>::value, A>::type
Foldable<_Maybe>::foldl1(function_t<A(Arg1, Arg2)> const&, Maybe<A> const& x)
{
    if (!x)
        throw maybe_nothing_error("foldl1");
    return x.value();
}

// foldr :: (a -> b -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
typename std::enable_if<is_same_as<Ret, B>::value, Ret>::type
Foldable<_Maybe>::foldr(function_t<Ret(A, B)> const& f, Ret const& z, Maybe<fdecay<A> > const& x){
    return x ? f(x.value(), z) : z;
}

// foldr1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
typename std::enable_if<is_same_as<A, Arg1>::value&& is_same_as<A, Arg2>::value, A>::type
Foldable<_Maybe>::foldr1(function_t<A(Arg1, Arg2)> const&, Maybe<A> const& x)
{
    if(!x)
        throw maybe_nothing_error("foldr1");
    return x.value();
}

// Traversable
// instance Traversable Maybe where
//    traverse :: Applicative f => (a -> f b) -> Maybe a -> f (Maybe b)
//    traverse _ Nothing = pure Nothing
//    traverse f (Just x) = Just <$> f x
template<typename AP, typename Arg>
typename std::enable_if<is_applicative_t<AP>::value, typeof_t<AP, Maybe<value_type_t<AP> > > >::type
Traversable<_Maybe>::traverse(function_t<AP(Arg)> const& f, Maybe<fdecay<Arg> > const& x){
    return x ? _(Just<value_type_t<AP> >) / f(x.value()) : Applicative_t<AP>::pure(Nothing<value_type_t<AP> >());
}

DEFAULT_SEQUENCEA_IMPL(Maybe, _Maybe)

// MonadError
// throwError :: e -> m a
// throwError () = Nothing
template<typename A>
Maybe<A> MonadError<_Maybe>::throwError(error_type<A> const&) {
    return Nothing<A>();
}

// catchError :: m a -> (e -> m a) -> m a
// catchError Nothing f = f ()
// catchError x       _ = x
template<typename A>
Maybe<A> MonadError<_Maybe>::catchError(Maybe<A> const& x, function_t<Maybe<A>(error_type<A> const&)> const& f) {
    return x ? x : f(error_type<A>());
}

// Maybe
DEFINE_FUNCTION_3(2, Maybe<T1>, maybe, T1 const&, default_value, function_t<T1(T0)> const&, f, Maybe<fdecay<T0> > const&, v,
    return Just(v ? f(v.value()) : default_value);)

template<typename T>
bool isJust(Maybe<T> const& value) {
    return (bool)value;
}

template<typename T>
bool isNothing(Maybe<T> const& value) {
    return !value;
}

template<typename T>
T fromJust(Maybe<T> const& value)
{
    if (!value)
        throw maybe_nothing_error("fromJust");
    return value.value();
}

DEFINE_FUNCTION_2(1, T0, fromMaybe, T0 const&, default_value, Maybe<T0> const&, value,
    return value ? value.value() : default_value;)

template<typename T0> T0 fromMaybe(f0<T0> const& f, Maybe<T0> const& value) {
    return value ? value.value() : *f;
}

_FUNCPROG_END

namespace std {

template<typename T>
ostream& operator<<(ostream& os, _FUNCPROG::Maybe<T> const& mv) {
    return mv ? os << "Just(" << mv.value() << ')' : os << "Nothing";
}

template<typename T>
wostream& operator<<(wostream& os, _FUNCPROG::Maybe<T> const& mv) {
    return mv ? os << L"Just(" << mv.value() << L')' : os << L"Nothing";
}

ostream& operator<<(ostream& os, _FUNCPROG::Maybe<string> const& mv) {
    return mv ? os << "Just(\"" << mv.value() << "\")" : os << "Nothing";
}

wostream& operator<<(wostream& os, _FUNCPROG::Maybe<wstring> const& mv) {
    return mv ? os << L"Just(\"" << mv.value() << L"\")" : os << L"Nothing";
}

ostream& operator<<(ostream& os, _FUNCPROG::Maybe<_FUNCPROG::f0<string> > const& mv) {
    return mv ? os << "Just(\"" << mv.value()() << "\")" : os << "Nothing";
}

wostream& operator<<(wostream& os, _FUNCPROG::Maybe<_FUNCPROG::f0<wstring> > const& mv) {
    return mv ? os << L"Just(\"" << mv.value()() << L"\")" : os << L"Nothing";
}

template<typename A>
std::ostream& operator<<(std::ostream& os, _FUNCPROG::EmptyData<A> const&) {
    return os << "()";
}

template<typename A>
std::wostream& operator<<(std::wostream& os, _FUNCPROG::EmptyData<A> const&) {
    return os << L"()";
}

} // namespace std
