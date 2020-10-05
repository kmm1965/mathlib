#pragma once

_FUNCPROG_BEGIN

// Functor
// <$> fmap :: Functor f => (a -> b) -> f a -> f b
template<typename Ret, typename Arg, typename... Args>
Identity<remove_f0_t<function_t<Ret(Args...)> > >
Functor<_Identity>::fmap(function_t<Ret(Arg, Args...)> const& f, Identity<fdecay<Arg> > const& v){
    return f << v.run();
}

// Applicative
template<typename T>
Identity<fdecay<T> > Applicative<_Identity>::pure(T const& x) {
    return x;
}

template<typename Ret, typename Arg, typename... Args>
Identity<remove_f0_t<function_t<Ret(Args...)> > >
Applicative<_Identity>::apply(Identity<function_t<Ret(Arg, Args...)> > const& f, Identity<fdecay<Arg> > const& v){
    return super::fmap(f.run(), v);
}

// Monad
IMPLEMENT_MRETURN(Identity, _Identity)

template<typename Ret, typename Arg, typename... Args>
remove_f0_t<function_t<Identity<Ret>(Args...)> >
Monad<_Identity>::mbind(Identity<fdecay<Arg> > const& m, function_t<Identity<Ret>(Arg, Args...)> const& f){
    return f << m.run();
}

// Foldable
DEFAULT_FOLDMAP_IMPL(Identity, _Identity)

// foldl :: (b -> a -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
typename std::enable_if<is_same_as<Ret, B>::value, Ret>::type
Foldable<_Identity>::foldl(function_t<Ret(B, A)> const& f, Ret const& z, Identity<fdecay<A> > const& x){
    return f(z, x.run());
}

// foldl1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
typename std::enable_if<is_same_as<A, Arg1>::value&& is_same_as<A, Arg2>::value, A>::type
Foldable<_Identity>::foldl1(function_t<A(Arg1, Arg2)> const&, Identity<A> const& x){
    return x.run();
}

// foldr :: (a -> b -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
typename std::enable_if<is_same_as<Ret, B>::value, Ret>::type
Foldable<_Identity>::foldr(function_t<Ret(A, B)> const& f, Ret const& z, Identity<fdecay<A> > const& x){
    return f(x.run(), z);
}

// foldr1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
typename std::enable_if<is_same_as<A, Arg1>::value&& is_same_as<A, Arg2>::value, A>::type
Foldable<_Identity>::foldr1(function_t<A(Arg1, Arg2)> const&, Identity<A> const& x){
    return x.run();
}

// Traversable
template<typename AP, typename Arg>
typename std::enable_if<is_applicative<AP>::value, typeof_t<AP, Identity<value_type_t<AP> > > >::type
Traversable<_Identity>::traverse(function_t<AP(Arg)> const& f, Identity<fdecay<Arg> > const& x){
    return _(Identity_<value_type_t<AP> >) / f(x.run());
}

DEFAULT_SEQUENCEA_IMPL(Identity, _Identity)

_FUNCPROG_END

namespace std {

template<typename T>
ostream& operator<<(ostream& os, _FUNCPROG::Identity<T> const& v) {
    return std::operator<<(os, "Identity(") << v.run() << ')';
}

template<typename T>
wostream& operator<<(wostream& os, _FUNCPROG::Identity<T> const& v) {
    return std::operator<<(os, L"Identity(") << v.run() << L')';
}

ostream& operator<<(ostream& os, _FUNCPROG::Identity<string> const& v) {
    return os << "Identity(\"" << v.run() << "\")";
}

wostream& operator<<(wostream& os, _FUNCPROG::Identity<wstring> const& v) {
    return os << L"Identity(\"" << v.run() << L"\")";
}

ostream& operator<<(ostream& os, _FUNCPROG::Identity<_FUNCPROG::f0<string> > const& v) {
    return os << "Identity(\"" << v.run() << "\")";
}

wostream& operator<<(wostream& os, _FUNCPROG::Identity<_FUNCPROG::f0<wstring> > const& v) {
    return os << L"Identity(\"" << v.run() << L"\")";
}

} // namespace std
