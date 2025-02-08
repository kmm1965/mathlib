#pragma once

#include "fwd/List_fwd.hpp"

_FUNCPROG_BEGIN

template<typename _M>
struct _Monoid // Default implementation of some functions
{
    // Default implementation of mappend
    template<typename A>
    static constexpr monoid_type<A> mappend(A const& x, A const& y);

    // Default implementation of mconcat
    // mconcat :: [a] -> a
    template<typename A>
    static constexpr monoid_type<A> mconcat(List<A> const& m);
};

_FUNCPROG_END
