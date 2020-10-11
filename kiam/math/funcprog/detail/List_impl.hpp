#pragma once

#include <vector>

#include "Monad_impl.hpp"
#include "map.hpp"

_FUNCPROG_BEGIN

class list_error : monad_error
{
public:
    list_error(const char *msg) : monad_error(msg) {}
};

class empty_list_error : list_error
{
public:
    empty_list_error(const char *msg) : list_error(msg) {}
};

// Functor
// <$> fmap :: Functor f => (a -> b) -> f a -> f b
template<typename Ret, typename Arg, typename... Args>
List<remove_f0_t<function_t<Ret(Args...)> > >
constexpr Functor<_List>::fmap(function_t<Ret(Arg, Args...)> const& f, List<fdecay<Arg> > const& l){
    return map(f, l);
}

// Applicative
template<typename T>
constexpr List<fdecay<T> > Applicative<_List>::pure(T const& x) {
    return { x };
}

template<typename Ret, typename Arg, typename... Args>
List<remove_f0_t<function_t<Ret(Args...)> > >
constexpr Applicative<_List>::apply(List<function_t<Ret(Arg, Args...)> > const& f, List<fdecay<Arg> > const& l)
{
    List<remove_f0_t<function_t<Ret(Args...)> > > result;
    for (function_t<Ret(Arg, Args...)> const& f_ : f)
        map_impl(result, f_, l);
    return result;
}

// Monad
IMPLEMENT_MRETURN(List, _List)

template<typename Ret, typename Arg, typename... Args>
remove_f0_t<function_t<List<Ret>(Args...)> >
constexpr Monad<_List>::mbind(List<fdecay<Arg> > const& l, function_t<List<Ret>(Arg, Args...)> const& f)
{
    return invoke_f0(_([l, f](Args... args) {
        List<Ret> result;
        for (fdecay<Arg> const& v : l) {
            const List<Ret> l2 = f(v, args...);
            result.insert(std::end(result), std::cbegin(l2), std::cend(l2));
        }
        return result;
    }));
}

// MonadPlus
IMPLEMENT_DEFAULT_MONADPLUS(List, _List)

// Alternative
template<typename A>
constexpr List<A> Alternative<_List>::empty() {
    return List<A>();
}

template<typename A>
constexpr List<A> Alternative<_List>::alt_op(List<A> const& l, List<A> const&r) {
    return l + r;
}

// Semigroup
template<typename A>
constexpr List<A> Semigroup<_List>::semigroup_op(List<A> const& x, List<A> const& y) {
    return x + y;
}

template<typename A>
constexpr List<A> Semigroup<_List>::stimes(int n, List<A> const& l)
{
    assert(n >= 0);
    if (n < 0) throw list_error("stimes: [], negative multiplier");
    const function_t<List<A>(int)> rep = [&l, &rep](int i) {
        return i == 0 ? List<A>() : l + rep(i - 1);
    };
    return rep(n);
}

// Monoid
template<typename A>
List<A> Monoid<_List>::mconcat(List<List<A> > const& ls)
{
    List<A> result;
    for (List<A> const& l : ls)
        result.insert(std::end(result), std::cbegin(l), std::cend(l));
    return result;
}

// Foldable
DEFAULT_FOLDMAP_IMPL(List, _List)

// foldl :: (b -> a -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
constexpr typename std::enable_if<is_same_as<Ret, B>::value, Ret>::type
Foldable<_List>::foldl(function_t<Ret(B, A)> const& f, Ret const& z, List<fdecay<A> > const& l){
    return null(l) ? z : foldl(f, f(z, head(l)), tail(l));
}

// foldl1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
constexpr typename std::enable_if<is_same_as<A, Arg1>::value&& is_same_as<A, Arg2>::value, A>::type
Foldable<_List>::foldl1(function_t<A(Arg1, Arg2)> const& f, List<A> const& l){
    checkEmptyList(l, "foldl1");
    return foldl(f, head(l), tail(l));
}

// foldr :: (a -> b -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B>
constexpr typename std::enable_if<is_same_as<Ret, B>::value, Ret>::type
Foldable<_List>::foldr(function_t<Ret(A, B)> const& f, Ret const& z, List<fdecay<A> > const& l){
    return null(l) ? z : f(head(l), foldr(f, z, tail(l)));
}

// foldr1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2>
constexpr typename std::enable_if<is_same_as<A, Arg1>::value&& is_same_as<A, Arg2>::value, A>::type
Foldable<_List>::foldr1(function_t<A(Arg1, Arg2)> const& f, List<A> const& l){
    checkEmptyList(l, "foldr1");
    return foldr(f, head(l), tail(l));
}

// Traversable
// instance Traversable [] where
//    traverse :: Applicative f => (a -> f b) -> List a -> f (List b)
//    traverse f = List.foldr cons_f (pure [])
//       where cons_f x ys = liftA2 (:) (f x) ys
template<typename AP, typename Arg>
constexpr applicative_type<AP, typeof_t<AP, List<value_type_t<AP> > > >
Traversable<_List>::traverse(function_t<AP(Arg)> const& f, List<fdecay<Arg> > const& l)
{
    using A = fdecay<Arg>;
    using B = value_type_t<AP>;
    using FB = typeof_t<AP, List<B> >;
    const function_t<FB(A const&, FB const&)> cons_f = [&f](A const& x, FB const& ys) {
        return liftA2(_(cons<B>), f(x), ys);
    };
    return Foldable<_List>::foldr(cons_f, Applicative_t<AP>::pure(List<B>()), l);
}

DEFAULT_SEQUENCEA_IMPL(List, _List)

// List
template<typename A>
constexpr void checkEmptyList(List<A> const& l, const char *msg)
{
    assert(!l.empty());
    if (l.empty())
        throw empty_list_error(msg);
}

template<typename T>
constexpr T const& head(List<T> const& l)
{
    checkEmptyList(l, "head");
    return l.front();
}

template<typename T>
constexpr List<T> tail(List<T> const& l)
{
    checkEmptyList(l, "tail");
    return List<T>(++std::cbegin(l), std::cend(l));
}

template<typename T>
constexpr T const& last(List<T> const& l)
{
    checkEmptyList(l, "last");
    return l.back();
}

template<typename T>
constexpr List<T> init(List<T> const& l)
{
    checkEmptyList(l, "init");
    const typename List<T>::const_iterator ibegin = std::cbegin(l), iend = std::cend(l);
    typename List<T>::const_iterator it = ibegin, it1;
    while (++(it1 = it) != iend) it = it1;
    return List<T>(ibegin, it);
}

template<typename T>
constexpr bool null(List<T> const& l) {
    return l.empty();
}

template<typename T>
constexpr int length(List<T> const& l) {
    return (int)l.size();
}

DEFINE_FUNCTION_2(1, List<fdecay<T0> >, filter, function_t<bool(T0)> const&, pred, List<fdecay<T0> > const&, l,
    List<fdecay<T0> > result;
    std::copy_if(std::cbegin(l), std::cend(l), std::back_inserter(result), pred);
    return result;)

DEFINE_FUNCTION_2(1, List<T0>, cons, T0 const&, value, List<T0> const&, l,
    return value >> l;)

DEFINE_FUNCTION_2(1, List<T0>, concat2, List<T0> const&, l1, List<T0> const&, l2,
    return l1 + l2;)

// List
template<typename A>
constexpr List<A> operator>>(A const& value, List<A> const& l)
{
    List<A> result(l);
    result.push_front(value);
    return result;
}

List<char> operator>>(char value, List<char> const& l){
    return std::string(1, value) + l;
}

List<wchar_t> operator>>(wchar_t value, List<wchar_t> const& l){
    return std::wstring(1, value) + l;
}

template<typename A>
constexpr List<A> operator<<(List<A> const& l, A const& value)
{
    List<A> result(l);
    result.push_back(value);
    return result;
}

template<typename A>
constexpr List<A> operator+(List<A> const&l1, List<A> const&l2)
{
    List<A> result(l1);
    result.insert(std::end(result), std::cbegin(l2), std::cend(l2));
    return result;
}

_FUNCPROG_END

namespace std {

template<typename A>
ostream& operator<<(ostream& os, _FUNCPROG::List<A> const& l)
{
    os << '[';
    bool first = true;
    for (A const& item : l) {
        if (first) first = false;
        else os << ',';
        os << item;
    }
    return os << ']';
}

template<typename A>
wostream& operator<<(wostream& os, _FUNCPROG::List<A> const& l)
{
    os << L'[';
    bool first = true;
    for (A const& item : l) {
        if (first) first = false;
        else os << L',';
        os << item;
    }
    return os << L']';
}

ostream& operator<<(ostream& os, _FUNCPROG::List<char> const& l){
    return os << '"' << (const string&) l << '"';
}

wostream& operator<<(wostream& os, _FUNCPROG::List<wchar_t> const& l){
    return os << L'"' << (const wstring&) l << L'"';
}

ostream& operator<<(ostream& os, _FUNCPROG::List<string> const& l)
{
    os << '[';
    bool first = true;
    for (const string &s : l) {
        if (first) first = false;
        else os << ',';
        os << '"' << s << '"';
    }
    return os << ']';
}

wostream& operator<<(wostream& os, _FUNCPROG::List<wstring> const& l)
{
    os << L'[';
    bool first = true;
    for (const wstring &s : l) {
        if (first) first = false;
        else os << L',';
        os << L'"' << s << L'"';
    }
    return os << L']';
}

ostream& operator<<(ostream& os, _FUNCPROG::List<_FUNCPROG::f0<string> > const& l)
{
    os << '[';
    bool first = true;
    for (_FUNCPROG::f0<string> const& f : l) {
        if (first) first = false;
        else os << ',';
        os << '"' << f() << '"';
    }
    return os << ']';
}

wostream& operator<<(wostream& os, _FUNCPROG::List<_FUNCPROG::f0<wstring> > const& l)
{
    os << '[';
    bool first = true;
    for (_FUNCPROG::f0<wstring> const& f : l) {
        if (first) first = false;
        else os << L',';
        os << L'"' << f() << L'"';
    }
    return os << L']';
}

template<typename A>
ostream& operator<<(ostream& os, _FUNCPROG::List<_FUNCPROG::f0<A> > const& l)
{
    os << '[';
    bool first = true;
    for (_FUNCPROG::f0<A> const& f : l) {
        if (first) first = false;
        else os << ',';
        os << f();
    }
    return os << ']';
}

template<typename A>
wostream& operator<<(wostream& os, _FUNCPROG::List<_FUNCPROG::f0<A> > const& l)
{
    os << L'[';
    bool first = true;
    for (_FUNCPROG::f0<A> const& f : l) {
        if (first) first = false;
        else os << L',';
        os << f();
    }
    return os << L']';
}

} // namespace std

#include "ListAPI.hpp"
