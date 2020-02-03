#pragma once

_FUNCPROG_BEGIN

/*
-- | 'and' returns the conjunction of a Boolean list.  For the result to be
-- 'True', the list must be finite; 'False', however, results from a 'False'
-- value at a finite index of a finite or infinite list.
and                     :: [Bool] -> Bool
#if defined(USE_REPORT_PRELUDE)
and                     =  foldr (&&) True
#else
and []          =  True
and (x:xs)      =  x && and xs
*/
inline bool _and(List<bool> const& l)
{
    /* return foldr(_(std::logical_and<bool>()), true, l); */
    for (bool b : l)
        if (!b) return false;
    return true;
}

inline bool _and_(List<bool> const& l)
{
    /* return foldr(_(std::logical_and<bool>()), true, l); */
    for (bool b : l)
        if (!b) return false;
    return true;
}

inline bool _or(List<bool> const& l)
{
    /* return foldr(_(std::logical_and<bool>()), true, l); */
    for (bool b : l)
        if (b) return true;
    return false;
}

inline bool _or_(List<bool> const& l)
{
    /* return foldr(_(std::logical_or<bool>()), false, l); */
    for (bool b : l)
        if (b) return true;
    return false;
}

DEFINE_FUNCTION_2(1, bool, any, function_t<bool(T0)> const&, p, List<fdecay<T0> > const&, l,
    return (_(_or_) & _map(p))(l);)

/*
-- | Applied to a predicate and a list, 'all' determines if all elements
-- of the list satisfy the predicate. For the result to be
-- 'True', the list must be finite; 'False', however, results from a 'False'
-- value for the predicate applied to an element at a finite index of a finite or infinite list.
all                     :: (a -> Bool) -> [a] -> Bool
#if defined(USE_REPORT_PRELUDE)
all p                   =  and . map p
#else
all _ []        =  True
all p (x:xs)    =  p x && all p xs
*/
DEFINE_FUNCTION_2(1, bool, all, function_t<bool(T0)> const&, p, List<fdecay<T0> > const&, l,
    return (_(_and_) & _map(p))(l);)

/*
-- | 'elem' is the list membership predicate, usually written in infix form,
-- e.g., @x \`elem\` xs@.  For the result to be
-- 'False', the list must be finite; 'True', however, results from an element
-- equal to @x@ found at a finite index of a finite or infinite list.
elem                    :: (Eq a) => a -> [a] -> Bool
#if defined(USE_REPORT_PRELUDE)
elem x                  =  any (== x)
#else
elem _ []       = False
elem x (y:ys)   = x==y || elem x ys
{-# NOINLINE [1] elem #-}
{-# RULES
"elem/build"    forall x (g :: forall b . Eq a => (a -> b -> b) -> b -> b)
   . elem x (build g) = g (\ y r -> (x == y) || r) False
 #-}
#endif
*/
// elem :: (Eq a) => a -> [a] -> Bool
// elem x = any(== x)
DEFINE_FUNCTION_2(1, bool, elem, T0 const&, v, List<T0> const&, l,
    return any(_eq(v), l);)

/*
-- | 'notElem' is the negation of 'elem'.
notElem                 :: (Eq a) => a -> [a] -> Bool
#if defined(USE_REPORT_PRELUDE)
notElem x               =  all (/= x)
#else
notElem _ []    =  True
notElem x (y:ys)=  x /= y && notElem x ys
{-# NOINLINE [1] notElem #-}
{-# RULES
"notElem/build" forall x (g :: forall b . Eq a => (a -> b -> b) -> b -> b)
   . notElem x (build g) = g (\ y r -> (x /= y) && r) True
 #-}
#endif
*/
// notElem                 :: (Eq a) = > a ->[a]->Bool
// notElem x = all(/= x)
DEFINE_FUNCTION_2(1, bool, notElem, T0 const&, v, List<T0> const&, l,
    return all(_neq(v), l);)

_FUNCPROG_END
