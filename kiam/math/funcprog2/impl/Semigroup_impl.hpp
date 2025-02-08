#pragma once

#include "../Semigroup.hpp"
#include "../fwd/Monoid_fwd.hpp"

_FUNCPROG2_BEGIN

//-- | This is a valid definition of 'stimes' for an idempotent 'Monoid'.
//--
//-- When @mappend x x = x@, this definition should be preferred, because it
//-- works in \(\mathcal{O}(1)\) rather than \(\mathcal{O}(\log n)\)
//stimesIdempotentMonoid :: (Integral b, Monoid a) => b -> a -> a
//stimesIdempotentMonoid n x = case compare n 0 of
//  LT -> errorWithoutStackTrace "stimesIdempotentMonoid: negative multiplier"
//  EQ -> mempty
//  GT -> x
template<typename _S>
template<typename A>
constexpr std::enable_if_t<_is_monoid_v<_S>, typeof_t<_S, A> >
_Semigroup<_S>::stimesIdempotentMonoid(int n, typeof_t<_S, A> const& x)
{
    assert(n >= 0);
    if (n < 0) throw std::runtime_error("stimes: positive multiplier expected");
    return n == 0 ? Monoid<_S>::template mempty<A>() : x;
}

//-- | This is a valid definition of 'stimes' for a 'Monoid'.
//--
//-- Unlike the default definition of 'stimes', it is defined for 0
//-- and so it should be preferred where possible.
//stimesMonoid :: (Integral b, Monoid a) => b -> a -> a
//stimesMonoid n x0 = case compare n 0 of
//  LT -> errorWithoutStackTrace "stimesMonoid: negative multiplier"
//  EQ -> mempty
//  GT -> f x0 n
//    where
//      f x y
//        | even y = f (x `mappend` x) (y `quot` 2)
//        | y == 1 = x
//        | otherwise = g (x `mappend` x) (y `quot` 2) x               -- See Note [Half of y - 1]
//      g x y z
//        | even y = g (x `mappend` x) (y `quot` 2) z
//        | y == 1 = x `mappend` z
//        | otherwise = g (x `mappend` x) (y `quot` 2) (x `mappend` z) -- See Note [Half of y - 1]
template<typename _S>
template<typename A>
constexpr monoid_type<typeof_t<_S, A> >
_Semigroup<_S>::stimes(int n, typeof_t<_S, A> const& x)
{
    assert(n >= 0);
    if (n < 0) throw std::runtime_error("stimes: positive multiplier expected");
    using SA = typeof_t<_S, A>;
    if (n == 0)
        return Monoid<_S>::template mempty<A>();
    auto const g = _([&g](SA const& x, int n, SA const& z){
        return
            n % 2 == 0 ? g(Monoid<_S>::mappend(x, x), n / 2, z) :
            n == 1 ? Monoid<_S>::mappend(x, z) :
            g(Monoid<_S>::mappend(x, x), n / 2, Monoid<_S>::mappend(x, z));
    });
    auto const f = _([&f, &g](SA const& x, int n){
        return
            n % 2 == 0 ? f(Monoid<_S>::mappend(x, x), n / 2) :
            n == 1 ? x : g(Monoid<_S>::mappend(x, x), n / 2, x);
    });
    return f(x, n);
}

//stimesDefault :: (Integral b, Semigroup a) => b -> a -> a
//stimesDefault y0 x0
//  | y0 <= 0   = errorWithoutStackTrace "stimes: positive multiplier expected"
//  | otherwise = f x0 y0
//  where
//    f x y
//      | even y = f (x <> x) (y `quot` 2)
//      | y == 1 = x
//      | otherwise = g (x <> x) (y `quot` 2) x        -- See Note [Half of y - 1]
//    g x y z
//      | even y = g (x <> x) (y `quot` 2) z
//      | y == 1 = x <> z
//      | otherwise = g (x <> x) (y `quot` 2) (x <> z) -- See Note [Half of y - 1]
template<typename _S>
template<typename A>
constexpr std::enable_if_t<!_is_monoid<_S>::value, typeof_t<_S, A> >
_Semigroup<_S>::stimes(int n, typeof_t<_S, A> const& x)
{
    assert(n > 0);
    if (n <= 0) throw std::runtime_error("stimes: positive multiplier expected");
    using SA = typeof_t<_S, A>;
    auto const g = _([&g](SA const& x, int n, SA const& z){
        return
            n % 2 == 0 ? g(x % x, n / 2, z) :
            n == 1 ? x % z :
            g(x % x, n / 2, x % z);
    });
    auto const f = _([&f, &g](SA const& x, int n){
        return
            n % 2 == 0 ? f(x % x, n / 2) :
            n == 1 ? x : g(x % x, n / 2, x);
    });
    return f(x, n);
}

template<typename S>
constexpr semigroup_type<S> semigroup_op(S const& x, S const& y){
    return Semigroup_t<S>::sg_op(x, y);
}

template<typename S>
constexpr semigroup_type<S> stimes(int n, S const& s){
    return Semigroup_t<S>::stimes(n, s);
}

_FUNCPROG2_END
