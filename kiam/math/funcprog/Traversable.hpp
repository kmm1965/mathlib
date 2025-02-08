#pragma once

#include "fwd/Traversable_fwd.hpp"

_FUNCPROG_BEGIN

template<typename _T>
struct _Traversable // Default implementation of some functions
{
    // sequenceA :: Applicative f => t (f a) -> f (t a)
    template<typename A> 
    static constexpr applicative_type<A, typeof_t<A, typeof_t<_T, value_type_t<A> > > >
    sequenceA(typeof_t<_T, A> const& x);
};

#define DECLARE_TRAVERSABLE_CLASS(T) \
    /* traverse :: Applicative f => (a -> f b) -> t a -> f (t b) */ \
    template<typename AP, typename Arg> \
    static constexpr applicative_type<AP, typeof_t<AP, T<value_type_t<AP> > > > \
    traverse(function_t<AP(Arg)> const& f, T<fdecay<Arg> > const& x);

_FUNCPROG_END
