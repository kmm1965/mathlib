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
inline bool and_(List<bool> const& l)
{
    /* return foldr(_(std::logical_and<bool>()), true, l); */
    for (bool b : l)
        if (!b) return false;
    return true;
}

inline bool or_(List<bool> const& l)
{
    /* return foldr(_(std::logical_and<bool>()), true, l); */
    for (bool b : l)
        if (b) return true;
    return false;
}

_FUNCPROG_END
