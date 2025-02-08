#pragma once

#include "fwd/List_fwd.hpp"
#include "Functor.hpp"
#include "Applicative.hpp"
#include "MonadFail.hpp"
#include "Monad.hpp"
//#include "Alternative.hpp"
//#include "MonadPlus.hpp"
#include "Semigroup.hpp"
#include "Monoid.hpp"
#include "Foldable.hpp"
#include "Traversable.hpp"
#include "Monad/MonadZip.hpp"
#include "detail/minmax.hpp"

_FUNCPROG2_BEGIN

class list_error : monad_error
{
public:
    list_error(const char* msg) : monad_error(msg){}
};

class empty_list_error : list_error
{
public:
    empty_list_error(const char* msg) : list_error(msg){}
};

struct _List
{
    using base_class = _List;

    template<typename A>
    using type = List<A>;
};

template<typename A>
struct List : list_t<A>, _List
{
    using value_type = A;
    using super = list_t<value_type>;

    List(){}
    List(std::initializer_list<value_type> const& il) : super(il){}
    List(List const& other) : super(other){}
    template<typename FuncImpl>
    List(f0<List, FuncImpl> const& f) : super(*f){}
    template<class Iter> List(Iter first, Iter last) : super(first, last){}
    List(size_t _Count, const value_type& _Val) : super(_Count, _Val){}
    List(A const& x) : super(1, x){}
};

template<>
struct List<char> : std::string, _List
{
    using value_type = char;
    using super = std::string;

    List(){}
    List(std::initializer_list<char> const& il) : super(il){}
    List(const char *value) : super(value){}
    List(super const& value) : super(value){}
    List(List const& other) : super(other){}
    template<typename FuncImpl>
    List(f0<List, FuncImpl> const& f) : super(*f){}
    template<class Iter> List(Iter first, Iter last) : super(first, last){}
    List(size_t _Count, char _Val) : super(_Count, _Val){}
    List(char x) : super(1, x){}
};

using String = List<char>;

inline bool sempty(String const& s){
    return s.empty();
}

template<>
struct List<wchar_t> : std::wstring, _List
{
    using value_type = char;
    using super = std::wstring;

    List(){}
    List(std::initializer_list<wchar_t> const& il) : super(il){}
    List(const wchar_t *value) : super(value){}
    List(super const& value) : super(value){}
    List(List const& other) : super(other){}
    template<typename FuncImpl>
    List(f0<List, FuncImpl> const& f) : super(*f){}
    template<class Iter> List(Iter first, Iter last) : super(first, last){}
    List(size_t _Count, char _Val) : super(_Count, _Val){}
    List(wchar_t x) : super(1, x){}
};

using wString = List<wchar_t>;

// Functor
template<>
struct Functor<_List> : _Functor<_List>
{
    // <$> fmap :: Functor f => (a -> b) -> f a -> f b
    template<typename Ret, typename Arg, typename... Args, typename FuncImpl>
    __DEVICE
    static constexpr auto fmap(function2<Ret(Arg, Args...), FuncImpl> const& f, List<fdecay<Arg> > const& v){
        return map(f, v);
    }
};

// Applicative
template<>
struct Applicative<_List> : Functor<_List>, _Applicative<_List>
{
    template<typename T>
    static constexpr List<fdecay<T> > pure(T const& x){
        return x;
    }

    template<typename Ret, typename Arg, typename FuncImpl>
    __DEVICE
    static constexpr List<Ret> apply(List<function2<Ret(Arg), FuncImpl> > const& f, List<fdecay<Arg> > const& v);
};

// MonadFail
template<>
struct MonadFail<_List>
{
    template<typename A = None>
    static constexpr List<A> fail(const char*){
        return List<A>();
    }
};

// Monad
template<>
struct Monad<_List> : Applicative<_List>, _Monad<_List>
{
    template<typename A>
    using liftM_type = List<A>;

    template<typename Ret, typename Arg, typename... Args, typename FuncImpl>
    static constexpr auto mbind(List<fdecay<Arg> > const& l, function2<List<Ret>(Arg, Args...), FuncImpl> const& f){
        return invoke_f0(_([l, f](Args... args) {
            List<Ret> result;
            for (fdecay<Arg> const& v : l) {
                const List<Ret> l2 = f(v, args...);
                result.insert(std::end(result), std::cbegin(l2), std::cend(l2));
            }
            return result;
        }));
    }
};

// Alternative
//template<>
//struct Alternative<_List> : _Alternative<_List>
//{
//    template<typename A>
//    static constexpr List<A> empty(){
//        return List<A>();
//    }
//
//    template<typename A>
//    static constexpr List<A> alt_op(List<A> const& l, List<A> const& r){
//        return l + r;
//    }
//};

// MonadPlus
//template<>
//struct MonadPlus<_List> : Monad<_List>, Alternative<_List>, _MonadPlus<_List>{};

// Semigroup
template<>
struct Semigroup<_List> : _Semigroup<_List>
{
    template<typename A>
    static constexpr List<A> sg_op(List<A> const& x, List<A> const& y){
        return x + y;
    }

    //stimesList  :: Integral b => b -> [a] -> [a]
    template<typename A>
    static constexpr List<A> stimes(int n, List<A> const& l);
};

// Monoid
template<>
struct Monoid<_List> : Semigroup<_List>, _Monoid<_List>
{
    template<typename A>
    static constexpr List<A> mempty(){
        return List<A>();
    }

    template<typename A>
    static constexpr List<A> mconcat(List<List<A> > const& ls);
};

// Foldable
template<>
struct Foldable<_List> : Monoid<_List>, _Foldable<_List>
{
    // foldl :: (b -> a -> b) -> b -> t a -> b
    template<typename Ret, typename A, typename B, typename FuncImpl>
    static constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
    foldl(function2<Ret(B, A), FuncImpl> const& f, Ret const& z, List<fdecay<A> > const& l);

    // foldl1 :: (a -> a -> a) -> t a -> a
    template<typename A, typename Arg1, typename Arg2, typename FuncImpl>
    static constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
    foldl1(function2<A(Arg1, Arg2), FuncImpl> const& f, List<A> const& l);

    // foldr :: (a -> b -> b) -> b -> t a -> b
    template<typename Ret, typename A, typename B, typename FuncImpl>
    static constexpr std::enable_if_t<is_same_as_v<Ret, B>, Ret>
    foldr(function2<Ret(A, B), FuncImpl> const& f, Ret const& z, List<fdecay<A> > const& l);

    // foldr1 :: (a -> a -> a) -> t a -> a
    template<typename A, typename Arg1, typename Arg2, typename FuncImpl>
    static constexpr std::enable_if_t<is_same_as_v<A, Arg1> && is_same_as_v<A, Arg2>, A>
    foldr1(function2<A(Arg1, Arg2), FuncImpl> const& f, List<A> const& l);

    template<typename A>
    static constexpr List<A> toList(List<A> const& l){
        return l;
    }

    template<typename T>
    static constexpr bool null(List<T> const& l){
        return l.empty();
    }

    template<typename T>
    static constexpr int length(List<T> const& l){
        return (int) l.size();
    }

    template<typename A>
    static constexpr A maximum(List<A> const& l);

    template<typename A>
    static constexpr A minimum(List<A> const& l);

    //-- | The 'sum' function computes the sum of a finite list of numbers.
    //sum :: (Num a) => [a] -> a
    template<typename A>
    static constexpr A sum(List<A> const& l);

    //-- | The 'product' function computes the product of a finite list of numbers.
    //product                 :: (Num a) => [a] -> a
    template<typename A>
    static constexpr A product(List<A> const& l);
};

// Traversable
template<>
struct Traversable<_List> : _Traversable<_List>
{
    // traverse :: Applicative f => (a -> f b) -> t a -> f (t b)
    template<typename AP, typename Arg, typename FuncImpl>
    static constexpr applicative_type<AP, typeof_t<AP, List<value_type_t<AP> > > >
    traverse(function2<AP(Arg), FuncImpl> const& f, List<fdecay<Arg> > const& l);
};

// MonadZip
template<>
struct MonadZip<_List> : _MonadZip<MonadZip<_List> >
{
    // mzip :: m a -> m b -> m (a,b)
    template<typename A, typename B>
    static constexpr List<pair_t<A, B> > mzip(List<A> const& ma, List<B> const& mb);

    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith = liftM2
    template<typename A, typename B, typename C, typename ArgA, typename ArgB, typename FuncImpl>
    static constexpr List<C> mzipWith(function2<C(ArgA, ArgB), FuncImpl> const& f, List<A> const& ma, List<B> const& mb);

    // munzip (Identity (a, b)) = (Identity a, Identity b)
    template<typename A, typename B>
    static constexpr pair_t<List<A>, List<B> > munzip(List<pair_t<A, B> > const& mab);
};

_FUNCPROG2_END
