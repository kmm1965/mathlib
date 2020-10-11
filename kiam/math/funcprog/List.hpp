#pragma once

#include "List_fwd.hpp"
#include "Monad.hpp"
#include "MonadPlus.hpp"
#include "Monoid.hpp"
#include "Foldable.hpp"
#include "Traversable.hpp"
#include "Monad/MonadZip.hpp"

_FUNCPROG_BEGIN

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
    List(f0<List> const& f) : super(*f) {}
    template<class Iter> List(Iter first, Iter last) : super(first, last){}
    List(size_t _Count, const value_type& _Val) : super(_Count, _Val){}
    List(A const& x) : super(1, x) {}
};

template<>
struct List<char> : std::string, _List
{
    using value_type = char;
    using super = std::string;

    List(){}
    List(std::initializer_list<char> const& il) : super(il) {}
    List(const char *value) : super(value){}
    List(super const& value) : super(value){}
    List(List const& other) : super(other){}
    List(f0<List> const& f) : super(*f) {}
    template<class Iter> List(Iter first, Iter last) : super(first, last){}
    List(size_t _Count, char _Val) : super(_Count, _Val) {}
    List(char x) : super(1, x) {}
};

using String = List<char>;

inline bool sempty(String const& s) {
    return s.empty();
}

template<>
struct List<wchar_t> : std::wstring, _List
{
    using value_type = char;
    using super = std::wstring;

    List(){}
    List(std::initializer_list<wchar_t> const& il) : super(il) {}
    List(const wchar_t *value) : super(value){}
    List(super const& value) : super(value){}
    List(List const& other) : super(other){}
    List(f0<List> const& f) : super(*f) {}
    template<class Iter> List(Iter first, Iter last) : super(first, last){}
    List(size_t _Count, char _Val) : super(_Count, _Val) {}
    List(wchar_t x) : super(1, x) {}
};

using wString = List<wchar_t>;

// Functor
IMPLEMENT_FUNCTOR(_List);

template<>
struct Functor<_List>
{
    DECLARE_FUNCTOR_CLASS(List)
};

// Applicative
IMPLEMENT_APPLICATIVE(_List);

template<>
struct Applicative<_List> : Functor<_List>
{
    typedef Functor<_List> super;

    DECLARE_APPLICATIVE_CLASS(List)
};

// Monad
IMPLEMENT_MONAD(_List);

template<>
struct Monad<_List> : Applicative<_List>
{
    typedef Applicative<_List> super;

    template<typename A>
    static constexpr List<fdecay<A> > mreturn(A const& x);

    template<typename Ret, typename Arg, typename... Args>
    static constexpr remove_f0_t<function_t<List<Ret>(Args...)> >
    mbind(List<fdecay<Arg> > const& m, function_t<List<Ret>(Arg, Args...)> const& f);

    template<typename A>
    using liftM_type = List<A>;
};

// Alternative
IMPLEMENT_ALTERNATIVE(_List);

template<>
struct Alternative<_List>
{
    DECLARE_ALTERNATIVE_CLASS(List)
};

// MonadPlus
IMPLEMENT_MONADPLUS(_List);

template<>
struct MonadPlus<_List> : Monad<_List>, Alternative<_List>
{
    using super = Alternative<_List>;

    DECLARE_MONADPLUS_CLASS(List)
};

// Semigroup
IMPLEMENT_SEMIGROUP(_List);

template<>
struct Semigroup<_List>
{
    DECLARE_SEMIGROUP_CLASS(List)
};

// Monoid
IMPLEMENT_MONOID(_List);

template<>
struct Monoid<_List> : _Monoid, Semigroup<_List>
{
    template<typename A>
    static List<A> mempty() {
        return List<A>();
    }

    template<typename A>
    static List<A> mconcat(List<List<A> > const& ls);
};

// Foldable
IMPLEMENT_FOLDABLE(_List);

template<>
struct Foldable<_List> : Monoid<_List>
{
    DECLARE_FOLDABLE_CLASS(List)
};

// Traversable
IMPLEMENT_TRAVERSABLE(_List);

template<>
struct Traversable<_List>
{
    DECLARE_TRAVERSABLE_CLASS(List)
};

// MonadZip
template<>
struct MonadZip<_List> : _MonadZip<MonadZip<_List> >
{
    // mzip :: m a -> m b -> m (a,b)
    template<typename A, typename B>
    static List<pair_t<A, B> > mzip(List<A> const& ma, List<B> const& mb){
        return zip(ma, mb);
    }

    // mzipWith :: (a -> b -> c) -> m a -> m b -> m c
    // mzipWith = liftM2
    template<typename A, typename B, typename C, typename ArgA, typename ArgB>
    static List<C> mzipWith(function_t<C(ArgA, ArgB)> const& f, List<A> const& ma, List<B> const& mb)
    {
        static_assert(is_same_as<ArgA, A>::value, "Should be the same");
        static_assert(is_same_as<ArgB, B>::value, "Should be the same");
        return zipWith(f, ma, mb);
    }

    // munzip (Identity (a, b)) = (Identity a, Identity b)
    template<typename A, typename B>
    static pair_t<List<A>, List<B> > munzip(List<pair_t<A, B> > const& mab){
        return unzip(mab);
    }
};

// List
template<typename A>
struct is_list : std::false_type {};

template<typename A>
struct is_list<List<A> > : std::true_type {};

template<typename L>
struct list_value_type {
    typedef void type;
};

template<typename A>
struct list_value_type<List<A> > {
    typedef A type;
};

/*
build   :: forall a. (forall b. (a -> b -> b) -> b -> b) -> [a]
{-# INLINE [1] build #-}
        -- The INLINE is important, even though build is tiny,
        -- because it prevents [] getting inlined in the version that
        -- appears in the interface file.  If [] *is* inlined, it
        -- won't match with [] appearing in rules in an importing module.
        --
        -- The "1" says to inline in phase 1

build g = g (:) []

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

_FUNCPROG_END

#include "detail/List_impl.hpp"
