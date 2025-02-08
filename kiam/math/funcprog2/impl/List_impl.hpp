#pragma once

#include "../List.hpp"

_FUNCPROG2_BEGIN

// List
template<typename A>
constexpr void checkEmptyList(List<A> const& l, const char* msg)
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

FUNCTION_TEMPLATE(2) constexpr List<fdecay<T0> > filter(FUNCTION2(bool(T0), T1) const& pred, List<fdecay<T0> > const& l)
{
    List<fdecay<T0> > result;
    std::copy_if(std::cbegin(l), std::cend(l), std::back_inserter(result), pred);
    return result;
}

FUNCTION_TEMPLATE(1) constexpr List<T0> cons(T0 const& value, List<T0> const& l){
    return value >> l;
}

FUNCTION_TEMPLATE(1) constexpr List<T0> concat2(List<T0> const& l1, List<T0> const& l2){
    return l1 + l2;
}

template<typename T>
struct concat2_ {
    constexpr List<T> operator()(List<T> const& l1, List<T> const& l2){
        return l1 + l2;
    }
};

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
constexpr List<A> operator+(List<A> const& l1, List<A> const& l2)
{
    List<A> result(l1);
    result.insert(std::end(result), std::cbegin(l2), std::cend(l2));
    return result;
}

/*
-- | A list producer that can be fused with 'foldr'.
-- This function is merely
--
-- >    build g = g (:) []
--
-- but GHC's simplifier will transform an expression of the form
-- @'foldr' k z ('build' g)@, which may arise after inlining, to @g k z@,
-- which avoids producing an intermediate list.

build   :: forall a. (forall b. (a -> b -> b) -> b -> b) -> [a]
{-# INLINE [1] build #-}
        -- The INLINE is important, even though build is tiny,
        -- because it prevents [] getting inlined in the version that
        -- appears in the interface file.  If [] *is* inlined, it
        -- won't match with [] appearing in rules in an importing module.
        --
        -- The "1" says to inline in phase 1

build g = g (:) []
*/
template<typename A, typename FuncImpl1, typename FuncImpl2>
constexpr List<A> build(function2<List<A>(function2<List<A>(A const&, List<A> const&), FuncImpl1> const&, List<A> const&), FuncImpl2> const& g){
    return g(_(cons<A>), List<A>());
}

/*
-- | A list producer that can be fused with 'foldr'.
-- This function is merely
--
-- >    augment g xs = g (:) xs
--
-- but GHC's simplifier will transform an expression of the form
-- @'foldr' k z ('augment' g xs)@, which may arise after inlining, to
-- @g k ('foldr' k z xs)@, which avoids producing an intermediate list.

augment :: forall a. (forall b. (a->b->b) -> b -> b) -> [a] -> [a]
{-# INLINE [1] augment #-}
augment g xs = g (:) xs
*/
template<typename A, typename B, typename FuncImpl1, typename FuncImpl2>
constexpr List<A> augment(function2<B(function2<B(A const&, B const&), FuncImpl1> const&, B const&), FuncImpl2> const& g, List<A> const& xs){
    return g(_(cons<A>), xs);
}

// Applicative
template<typename Ret, typename Arg, typename FuncImpl>
__DEVICE
constexpr List<Ret> Applicative<_List>::apply(List<function2<Ret(Arg), FuncImpl> > const& f, List<fdecay<Arg> > const& v)
{
    List<Ret> result;
    for(auto const& f_ : f)
        map_impl(result, f_, v);
    return result;
}

// Semigroup
//stimesList  :: Integral b => b -> [a] -> [a]
//stimesList n x
//  | n < 0 = errorWithoutStackTrace "stimes: [], negative multiplier"
//  | otherwise = rep n
//  where
//    rep 0 = []
//    rep i = x ++ rep (i - 1)
template<typename A>
constexpr List<A>
Semigroup<_List>::stimes(int n, List<A> const& l)
{
    assert(n >= 0);
    if (n < 0) throw list_error("stimes: [], negative multiplier");
    auto const rep = [&l, &rep](int i){
        return i == 0 ? List<A>() : l + rep(i - 1);
    };
    return rep(n);
}

// Monoid
template<typename A>
constexpr List<A>
Monoid<_List>::mconcat(List<List<A> > const& ls)
{
    List<A> result;
    for (List<A> const& l : ls)
        result.insert(std::end(result), std::cbegin(l), std::cend(l));
    return result;
}

// Foldable
// foldl :: (b -> a -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B, typename FuncImpl>
constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
Foldable<_List>::foldl(function2<Ret(B, A), FuncImpl> const& f, Ret const& z, List<fdecay<A> > const& l){
    return null(l) ? z : foldl(f, f(z, head(l)), tail(l));
}

// foldl1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2, typename FuncImpl>
constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
Foldable<_List>::foldl1(function2<A(Arg1, Arg2), FuncImpl> const& f, List<A> const& l){
    checkEmptyList(l, "foldl1");
    return foldl(f, head(l), tail(l));
}

// foldr :: (a -> b -> b) -> b -> t a -> b
template<typename Ret, typename A, typename B, typename FuncImpl>
constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
Foldable<_List>::foldr(function2<Ret(A, B), FuncImpl> const& f, Ret const& z, List<fdecay<A> > const& l){
    return null(l) ? z : f(head(l), foldr(f, z, tail(l)));
}

// foldr1 :: (a -> a -> a) -> t a -> a
template<typename A, typename Arg1, typename Arg2, typename FuncImpl>
constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
Foldable<_List>::foldr1(function2<A(Arg1, Arg2), FuncImpl> const& f, List<A> const& l){
    checkEmptyList(l, "foldr1");
    return foldr(f, head(l), tail(l));
}

// maximum xs = foldl1 max xs
template<typename A>
constexpr A Foldable<_List>::maximum(List<A> const& l){
    checkEmptyList(l, "maximum");
    return foldl1(max<A>(), l);
}

// minimum xs = foldl1 min xs
template<typename A>
constexpr A Foldable<_List>::minimum(List<A> const& l){
    checkEmptyList(l, "minimum");
    return foldl1(min<A>(), l);
}

//-- | The 'sum' function computes the sum of a finite list of numbers.
//sum :: (Num a) => [a] -> a
//sum =  foldl (+) 0
template<typename A>
constexpr A Foldable<_List>::sum(List<A> const& l){
    return foldl(_(std::plus<A>()), A(), l);
}

//-- | The 'product' function computes the product of a finite list of numbers.
//product                 :: (Num a) => [a] -> a
//product                 =  foldl (*) 1
template<typename A>
constexpr A Foldable<_List>::product(List<A> const& l){
    return foldl(_(std::multiplies<A>()), A(1), l);
}

// Traversable
// traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
// instance Traversable [] where
//    traverse :: Applicative f => (a -> f b) -> List a -> f (List b)
//    traverse f = List.foldr cons_f (pure [])
//       where cons_f x ys = liftA2 (:) (f x) ys
template<typename AP, typename Arg, typename FuncImpl>
constexpr applicative_type<AP, typeof_t<AP, List<value_type_t<AP> > > >
Traversable<_List>::traverse(function2<AP(Arg), FuncImpl> const& f, List<fdecay<Arg> > const& l)
{
    using A = fdecay<Arg>;
    using B = value_type_t<AP>;
    using FB = typeof_t<AP, List<B> >;
    auto const cons_f = _([&f](A const& x, FB const& ys){
        return Applicative_t<AP>::liftA2(_(cons<B>))(f(x), ys);
    });
    return Foldable<_List>::foldr(cons_f, Applicative_t<AP>::pure(List<B>()), l);
}

// MonadZip
// mzip :: m a -> m b -> m (a,b)
template<typename A, typename B>
constexpr List<pair_t<A, B> >
MonadZip<_List>::mzip(List<A> const& ma, List<B> const& mb){
    return zip(ma, mb);
}

// mzipWith :: (a -> b -> c) -> m a -> m b -> m c
// mzipWith = liftM2
template<typename A, typename B, typename C, typename ArgA, typename ArgB, typename FuncImpl>
constexpr List<C>
MonadZip<_List>::mzipWith(function2<C(ArgA, ArgB), FuncImpl> const& f, List<A> const& ma, List<B> const& mb)
{
    static_assert(is_same_as_v<ArgA, A>, "Should be the same");
    static_assert(is_same_as_v<ArgB, B>, "Should be the same");
    return zipWith(f, ma, mb);
}

// munzip (Identity (a, b)) = (Identity a, Identity b)
template<typename A, typename B>
constexpr pair_t<List<A>, List<B> >
MonadZip<_List>::munzip(List<pair_t<A, B> > const& mab){
    return unzip(mab);
}

_FUNCPROG2_END

namespace std {

template<typename A>
ostream& operator<<(ostream& os, _FUNCPROG2::List<A> const& l)
{
    os << '[';
    bool first = true;
    for (A const& item : l){
        if (first) first = false;
        else os << ',';
        os << item;
    }
    return os << ']';
}

template<typename A>
wostream& operator<<(wostream& os, _FUNCPROG2::List<A> const& l)
{
    os << L'[';
    bool first = true;
    for (A const& item : l){
        if (first) first = false;
        else os << L',';
        os << item;
    }
    return os << L']';
}

template<typename A, typename FuncImpl>
ostream& operator<<(ostream& os, _FUNCPROG2::List<_FUNCPROG2::f0<A, FuncImpl> > const& lf)
{
    os << '[';
    bool first = true;
    for(auto const& f : lf){
        if (first) first = false;
        else os << ',';
        os << f();
    }
    return os << ']';
}

template<typename A, typename FuncImpl>
wostream& operator<<(wostream& os, _FUNCPROG2::List<_FUNCPROG2::f0<A, FuncImpl> > const& lf)
{
    os << L'[';
    bool first = true;
    for(auto const& f : lf){
        if (first) first = false;
        else os << L',';
        os << f();
    }
    return os << L']';
}

inline ostream& operator<<(ostream& os, _FUNCPROG2::String const& s){
    return os << '"' << (const string&) s << '"';
}

inline wostream& operator<<(wostream& os, _FUNCPROG2::wString const& s){
    return os << L'"' << (const wstring&) s << L'"';
}

inline ostream& operator<<(ostream& os, _FUNCPROG2::List<string> const& ls)
{
    os << '[';
    bool first = true;
    for (string const& s : ls){
        if (first) first = false;
        else os << ',';
        os << '"' << s << '"';
    }
    return os << ']';
}

inline wostream& operator<<(wostream& os, _FUNCPROG2::List<wstring> const& ls)
{
    os << L'[';
    bool first = true;
    for (wstring const& s : ls){
        if (first) first = false;
        else os << L',';
        os << L'"' << s << L'"';
    }
    return os << L']';
}

template<typename FuncImpl>
inline ostream& operator<<(ostream& os, _FUNCPROG2::List<_FUNCPROG2::f0<string, FuncImpl> > const& lf)
{
    os << '[';
    bool first = true;
    for(auto const& f : lf){
        if (first) first = false;
        else os << ',';
        os << '"' << f() << '"';
    }
    return os << ']';
}

template<typename FuncImpl>
inline wostream& operator<<(wostream& os, _FUNCPROG2::List<_FUNCPROG2::f0<wstring, FuncImpl> > const& lf)
{
    os << '[';
    bool first = true;
    for(auto const& f : lf){
        if (first) first = false;
        else os << L',';
        os << L'"' << f() << L'"';
    }
    return os << L']';
}

} // namespace std

#include "ListAPI_impl.hpp"
