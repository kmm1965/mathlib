#pragma once

#include "../Identity.hpp"

_FUNCPROG_BEGIN

template<typename T>
constexpr T runIdentity(Identity<T> const& x){
    return x.run();
}

// Constructor
template<typename T>
constexpr Identity<T> Identity_(T const& value){
    return value;
}

template<typename T>
constexpr Identity<T> Identity_f(f0<T> const& fvalue){
    return fvalue;
}

// Semigroup
template<typename A>
constexpr semigroup_type<A, Identity<A> >
Semigroup<_Identity>::sg_op(Identity<A> const& x, Identity<A> const& y){
    return Semigroup_t<A>::sg_op(x.run(), y.run());
}

template<typename A>
constexpr semigroup_type<A, Identity<A> >
Semigroup<_Identity>::stimes(int n, Identity<A> const& m){
    return Semigroup_t<A>::stimes(n, m.run());
}

// Moniod
template<typename A>
constexpr monoid_type<A, Identity<A> >
Monoid<_Identity>::mempty(){
    return Monoid_t<A>::mempty();
}

// Foldable
// foldl :: (b -> a -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
Foldable<_Identity>::foldl(function_t<Ret(B, A)> const& f, Ret const& z, Identity<fdecay<A> > const& x){
    return f(z, x.run());
}

// foldl1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
Foldable<_Identity>::foldl1(function_t<A(Arg1, Arg2)> const&, Identity<A> const& x){
    return x.run();
}

// foldr :: (a -> b -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
Foldable<_Identity>::foldr(function_t<Ret(A, B)> const& f, Ret const& z, Identity<fdecay<A> > const& x){
    return f(x.run(), z);
}

// foldr1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
Foldable<_Identity>::foldr1(function_t<A(Arg1, Arg2)> const&, Identity<A> const& x){
    return x.run();
}

// Traversable
// traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
template<typename AP, typename Arg>
constexpr applicative_type<AP, typeof_t<AP, Identity<value_type_t<AP> > > >
Traversable<_Identity>::traverse(function_t<AP(Arg)> const& f, Identity<fdecay<Arg> > const& x){
    return Functor_t<AP>::fmap(_(Identity_<value_type_t<AP> >), f(x.run()));
}

// MonadZip
// mzipWith :: (a -> b -> c) -> m a -> m b -> m c
// mzipWith = liftM2
template<typename A, typename B, typename C, typename ArgA, typename ArgB>
constexpr Identity<C>
MonadZip<_Identity>::mzipWith(function_t<C(ArgA, ArgB)> const& f, Identity<A> const& ma, Identity<B> const& mb)
{
    static_assert(is_same_as_v<ArgA, A>, "Should be the same");
    static_assert(is_same_as_v<ArgB, B>, "Should be the same");
    return Monad<_Identity>::liftM2(f)(ma, mb);
}

// munzip (Identity (a, b)) = (Identity a, Identity b)
template<typename A, typename B>
constexpr pair_t<Identity<A>, Identity<B> >
MonadZip<_Identity>::munzip(Identity<pair_t<A, B> > const& mab)
{
    auto const& [a, b] = mab.run();
    return std::make_pair(Identity_(a), Identity_(b));
}

_FUNCPROG_END

namespace std {

    template<typename T>
    ostream& operator<<(ostream& os, _FUNCPROG::Identity<T> const& v){
        return std::operator<<(os, "Identity(") << v.run() << ')';
    }

    template<typename T>
    wostream& operator<<(wostream& os, _FUNCPROG::Identity<T> const& v){
        return std::operator<<(os, L"Identity(") << v.run() << L')';
    }

    ostream& operator<<(ostream& os, _FUNCPROG::Identity<string> const& v){
        return os << "Identity(\"" << v.run() << "\")";
    }

    wostream& operator<<(wostream& os, _FUNCPROG::Identity<wstring> const& v){
        return os << L"Identity(\"" << v.run() << L"\")";
    }

    ostream& operator<<(ostream& os, _FUNCPROG::Identity<_FUNCPROG::f0<string> > const& v){
        return os << "Identity(\"" << v.run() << "\")";
    }

    wostream& operator<<(wostream& os, _FUNCPROG::Identity<_FUNCPROG::f0<wstring> > const& v){
        return os << L"Identity(\"" << v.run() << L"\")";
    }

} // namespace std
