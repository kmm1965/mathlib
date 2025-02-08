#pragma once

_FUNCPROG_BEGIN

/*
-- | 'scanl' is similar to 'foldl', but returns a list of successive
-- reduced values from the left:
--
-- > scanl f z [x1, x2, ...] == [z, z `f` x1, (z `f` x1) `f` x2, ...]
--
-- Note that
--
-- > last (scanl f z xs) == foldl f z xs.

-- This peculiar arrangement is necessary to prevent scanl being rewritten in
-- its own right-hand side.
{-# NOINLINE [1] scanl #-}
scanl                   :: (b -> a -> b) -> b -> [a] -> [b]
scanl                   = scanlGo
where
scanlGo           :: (b -> a -> b) -> b -> [a] -> [b]
scanlGo f q ls    = q : (case ls of
[]   -> []
x:xs -> scanlGo f (f q x) xs)
*/
DECLARE_FUNCTION_3(3, List<T0>, scanl, function_t<T0(T1, T2)> const&, T0 const&, List<fdecay<T2> > const&)
FUNCTION_TEMPLATE(3) constexpr List<T0> scanl(function_t<T0(T1, T2)> const& f, T0 const& q, List<fdecay<T2> > const& l)
{
    static_assert(is_same_as_v<T0, T1>, "Should be the same");
    List<T0> result;
    result.push_back(q);
    T0 q1 = q;
    for (T2 const& x : l)
        result.push_back(q1 = f(q1, x));
    return result;
}

DECLARE_FUNCTION_2(3, List<T0>, scanl1, function_t<T0(T1, T2)> const&, List<T0> const&)

FUNCTION_TEMPLATE(3) constexpr List<T0> scanl1(function_t<T0(T1, T2)> const& f, List<T0> const& l)
{
    static_assert(is_same_as_v<T0, T1>, "Should be the same");
    static_assert(is_same_as_v<T0, T2>, "Should be the same");
    return null(l) ? List<T0>() : scanl(f, head(l), tail(l));
}

/*
-- | 'scanr' is the right-to-left dual of 'scanl'.
-- Note that
--
-- > head (scanr f z xs) == foldr f z xs.
scanr                   :: (a -> b -> b) -> b -> [a] -> [b]
scanr _ q0 []           =  [q0]
scanr f q0 (x:xs)       =  f x q : qs
                           where qs@(q:_) = scanr f q0 xs
*/
DECLARE_FUNCTION_3(3, List<T0>, scanr, function_t<T0(T1, T2)> const&, T0 const&, List<fdecay<T1> > const&)
FUNCTION_TEMPLATE(3) constexpr List<T0> scanr(function_t<T0(T1, T2)> const& f, T0 const& q, List<fdecay<T1> > const& l)
{
    static_assert(is_same_as_v<T0, T2>, "Should be the same");
    if (null(l))
        return List<T0>({ q });
    else {
        const List<fdecay<T0> > l2 = scanr(f, q, tail(l));
        return f(head(l), head(l2)) >> l2;
    }
}

DECLARE_FUNCTION_2(3, List<T0>, scanr1, function_t<T0(T1, T2)> const&, List<T0> const&)
FUNCTION_TEMPLATE(3) constexpr List<T0> scanr1(function_t<T0(T1, T2)> const& f, List<T0> const& l)
{
    static_assert(is_same_as_v<T0, T1>, "Should be the same");
    static_assert(is_same_as_v<T0, T2>, "Should be the same");
    if (null(l))
        return List<T0>();
    else if (length(l) == 1)
        return List<T0>({ head(l) });
    else {
        const List<T0> l2 = scanr1(f, tail(l));
        return f(head(l), head(l2)) >> l2;
    }
}

_FUNCPROG_END
